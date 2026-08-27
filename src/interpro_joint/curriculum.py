"""Curriculum for joint InterPro training."""

from __future__ import annotations
import random
from dataclasses import asdict, dataclass
from typing import Literal


TaskName = Literal["stage1", "stage2"]
PhaseName = Literal[
    "stage1_only",
    "stage1_three_to_one",
    "equal_mix",
]


@dataclass(frozen=True)
class CurriculumDecision:
    global_optimizer_step: int
    total_optimizer_steps: int
    phase: PhaseName
    phase_step: int
    phase_length: int
    task: TaskName
    stage1_probability: float
    stage2_probability: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CurriculumSummary:
    total_optimizer_steps: int

    stage1_only_steps: int
    three_to_one_steps: int
    equal_mix_steps: int

    stage1_steps: int
    stage2_steps: int

    stage1_fraction: float
    stage2_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class JointCurriculum:
    """
    Phase 1, first 15%: Stage 1 only.

    Phase 2, next 20%: Exactly 3 Stage 1 optimizer steps for each Stage 2 step.

    Phase 3, final 65%: Exactly equal Stage 1 and Stage 2 optimizer steps.

    Task order is deterministically shuffled inside each complete
    ratio block. Every DDP rank selects the same
    task without communication.

    """

    TOTAL_STEP_MULTIPLE = 40

    def __init__(
        self,
        total_optimizer_steps: int,
        seed: int = 47,
    ) -> None:
        if total_optimizer_steps <= 0:
            raise ValueError(
                "total_optimizer_steps must be positive"
            )

        if (
            total_optimizer_steps
            % self.TOTAL_STEP_MULTIPLE
            != 0
        ):
            raise ValueError(
                "total_optimizer_steps must be divisible by "
                f"{self.TOTAL_STEP_MULTIPLE}; observed "
                f"{total_optimizer_steps}"
            )

        self.total_optimizer_steps = int(
            total_optimizer_steps
        )

        self.seed = int(seed)

        units = (
            self.total_optimizer_steps
            // self.TOTAL_STEP_MULTIPLE
        )

        self.stage1_only_steps = 6 * units
        self.three_to_one_steps = 8 * units
        self.equal_mix_steps = 26 * units

        self.stage1_only_end = (
            self.stage1_only_steps
        )

        self.three_to_one_end = (
            self.stage1_only_steps
            + self.three_to_one_steps
        )

        if (
            self.stage1_only_steps
            + self.three_to_one_steps
            + self.equal_mix_steps
            != self.total_optimizer_steps
        ):
            raise RuntimeError(
                "Curriculum phase lengths do not sum "
                "to total_optimizer_steps"
            )

        if self.three_to_one_steps % 4 != 0:
            raise RuntimeError(
                "The 3:1 phase is not divisible by four"
            )

        if self.equal_mix_steps % 2 != 0:
            raise RuntimeError(
                "The equal phase is not divisible by two"
            )

    @staticmethod
    def _shuffled_block(
        *,
        template: tuple[TaskName, ...],
        seed: int,
    ) -> tuple[TaskName, ...]:
        block = list(template)

        random.Random(seed).shuffle(block)

        return tuple(block)

    def _three_to_one_task(
        self,
        phase_step: int,
    ) -> TaskName:
        block_size = 4
        block_index = phase_step // block_size
        position = phase_step % block_size

        block = self._shuffled_block(
            template=(
                "stage1",
                "stage1",
                "stage1",
                "stage2",
            ),
            seed=(
                self.seed
                + 1_000_003
                + block_index
            ),
        )

        return block[position]

    def _equal_mix_task(
        self,
        phase_step: int,
    ) -> TaskName:
        block_size = 2
        block_index = phase_step // block_size
        position = phase_step % block_size

        block = self._shuffled_block(
            template=(
                "stage1",
                "stage2",
            ),
            seed=(
                self.seed
                + 2000003
                + block_index
            ),
        )

        return block[position]

    def decision(
        self,
        global_optimizer_step: int,
    ) -> CurriculumDecision:
        """
        Return the task for one optimizer step.

        All microbatches in the same step use the returned task.
        """
        step = int(global_optimizer_step)

        if not 0 <= step < self.total_optimizer_steps:
            raise IndexError(
                "global_optimizer_step must be in "
                f"[0, {self.total_optimizer_steps}); "
                f"observed {step}"
            )

        if step < self.stage1_only_end:
            return CurriculumDecision(
                global_optimizer_step=step,
                total_optimizer_steps=(
                    self.total_optimizer_steps
                ),
                phase="stage1_only",
                phase_step=step,
                phase_length=(
                    self.stage1_only_steps
                ),
                task="stage1",
                stage1_probability=1.0,
                stage2_probability=0.0,
            )

        if step < self.three_to_one_end:
            phase_step = (
                step
                - self.stage1_only_end
            )

            return CurriculumDecision(
                global_optimizer_step=step,
                total_optimizer_steps=(
                    self.total_optimizer_steps
                ),
                phase="stage1_three_to_one",
                phase_step=phase_step,
                phase_length=(
                    self.three_to_one_steps
                ),
                task=self._three_to_one_task(
                    phase_step
                ),
                stage1_probability=0.75,
                stage2_probability=0.25,
            )

        phase_step = (
            step
            - self.three_to_one_end
        )

        return CurriculumDecision(
            global_optimizer_step=step,
            total_optimizer_steps=(
                self.total_optimizer_steps
            ),
            phase="equal_mix",
            phase_step=phase_step,
            phase_length=self.equal_mix_steps,
            task=self._equal_mix_task(
                phase_step
            ),
            stage1_probability=0.5,
            stage2_probability=0.5,
        )

    def task(
        self,
        global_optimizer_step: int,
    ) -> TaskName:
        return self.decision(
            global_optimizer_step
        ).task

    def summarize(self) -> CurriculumSummary:
        stage1_steps = 0
        stage2_steps = 0

        for step in range(
            self.total_optimizer_steps
        ):
            task = self.task(step)

            if task == "stage1":
                stage1_steps += 1
            elif task == "stage2":
                stage2_steps += 1
            else:
                raise RuntimeError(
                    f"Unexpected curriculum task: {task!r}"
                )

        if (
            stage1_steps + stage2_steps
            != self.total_optimizer_steps
        ):
            raise RuntimeError(
                "Curriculum task counts do not sum "
                "to total steps"
            )

        return CurriculumSummary(
            total_optimizer_steps=(
                self.total_optimizer_steps
            ),
            stage1_only_steps=(
                self.stage1_only_steps
            ),
            three_to_one_steps=(
                self.three_to_one_steps
            ),
            equal_mix_steps=(
                self.equal_mix_steps
            ),
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            stage1_fraction=(
                stage1_steps
                / self.total_optimizer_steps
            ),
            stage2_fraction=(
                stage2_steps
                / self.total_optimizer_steps
            ),
        )
