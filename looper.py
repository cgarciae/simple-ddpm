from dataclasses import dataclass, field
from datetime import datetime, timedelta
import functools
from re import U
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import inspect
from clu.periodic_actions import PeriodicAction
from pkbar import Kbar
from tqdm import tqdm
from flax.struct import PyTreeNode
import jax


class Elapsed(PyTreeNode):
    steps: int
    samples: int
    date: float
    _date_start: float

    @property
    def time(self) -> float:
        return self.date - self._date_start

    @classmethod
    def create(cls, steps: int = 0, samples: int = 0) -> "Elapsed":
        now = datetime.now().timestamp()
        return cls(steps=steps, samples=samples, _date_start=now, date=now)

    def update(self, batch_size: int) -> "Elapsed":
        return self.replace(
            steps=self.steps + 1,
            samples=self.samples + batch_size,
            date=datetime.now().timestamp(),
        )

    def advance_time(self) -> "Elapsed":
        return self.replace(date=datetime.now().timestamp())


class Period:
    def __init__(
        self,
        steps: Union[int, None] = None,
        samples: Union[int, None] = None,
        time: Union[timedelta, float, int, None] = None,
        date: Union[datetime, float, None] = None,
    ):
        if all(x is None for x in [steps, samples, time, date]):
            raise ValueError("At least one duration parameter must be specified.")

        self.steps = steps
        self.samples = samples
        self.time = time.total_seconds() if isinstance(time, timedelta) else time
        self.date = date.timestamp() if isinstance(date, datetime) else date

    def __repr__(self) -> str:
        params_repr = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items() if v is not None
        )
        return f"Duration({params_repr})"


State = Any
Batch = Any
Step = int
Logs = Dict[str, Any]
LogHistory = List[Logs]

Schedule = Callable[[Elapsed], bool]
Callback = Callable[[State, Batch, Elapsed, "Loop"], Tuple[Logs, State]]
InputCallable = Union[Callable, PeriodicAction]


def create_callback(f: InputCallable) -> Callback:

    if isinstance(f, PeriodicAction):

        @functools.wraps(f)
        def wrapper(state: State, batch: Batch, elapsed: Elapsed, loop: Loop):
            f(elapsed.steps)

    else:
        sig = inspect.signature(f)
        params = sig.parameters

        @functools.wraps(f)
        def wrapper(state: State, batch: Batch, elapsed: Elapsed, loop: Loop):
            # maybe inject logs and history
            args = [state, batch, elapsed]
            kwargs = {}
            if "loop" in params:
                kwargs["loop"] = loop

            return f(*args, **kwargs)

    return wrapper


# ---------------------------------------
# loops
# ---------------------------------------
def get_batch_size(batch: Batch) -> int:
    def get_size(sizes, x):
        sizes.add(x.shape[0])
        return sizes

    sizes = jax.tree_util.tree_reduce(get_size, batch, set())
    if len(sizes) != 1:
        raise ValueError("Batch size must be the same for all elements in the batch.")
    return sizes.pop()


class Loop:
    def __init__(
        self,
        state: State = None,
        initial_steps: int = 0,
        initial_samples: int = 0,
        logs: Optional[Logs] = None,
        history: Optional[LogHistory] = None,
    ):
        self.state = state
        self.initial_steps = initial_steps
        self.initial_samples = initial_samples
        self.logs = logs
        self.history = history or []

    def run(
        self,
        dataset,
        state: State,
        schedule_callbacks: Dict[Schedule, Union[InputCallable, List[InputCallable]]],
        batch_size: Optional[int] = None,
        total: Union[Period, int, None] = None,
    ) -> State:
        time_start = datetime.now().timestamp()

        elapsed = Elapsed.create(steps=self.initial_steps, samples=self.initial_samples)

        if isinstance(total, int):
            total = Period(steps=total)
        try:
            self.state = state
            schedule_callbacks_: Dict[Schedule, List[Callback]] = {
                schedule: [
                    create_callback(callback)
                    for callback in (
                        callbacks if isinstance(callbacks, list) else [callbacks]
                    )
                ]
                for schedule, callbacks in schedule_callbacks.items()
            }

            for i, batch in enumerate(dataset):
                if batch_size is None:
                    batch_size = get_batch_size(batch)

                self.logs = {}

                for schedule, callbacks in schedule_callbacks_.items():
                    if schedule(elapsed):
                        for callback in callbacks:
                            elapsed = elapsed.advance_time()
                            output = callback(self.state, batch, elapsed, self)
                            if output is not None:
                                logs, self.state = output
                                if logs is not None:
                                    self.logs.update(logs)

                elapsed = elapsed.advance_time()
                self.logs["elapsed"] = elapsed
                self.history.append(self.logs)
                elapsed = elapsed.update(batch_size)

                if total is not None:
                    if total.steps is not None and elapsed.steps >= total.steps:
                        break
                    elif total.samples is not None and elapsed.samples >= total.samples:
                        break
                    elif total.time is not None and elapsed.time >= total.time:
                        break
                    elif total.date is not None and elapsed.date >= total.date:
                        break

            return self.state
        finally:
            self.state = None
            self.logs = None


# ---------------------------------------
# schedules
# ---------------------------------------


class every:
    def __init__(
        self,
        steps: Union[int, None] = None,
        samples: Union[int, None] = None,
        time: Union[timedelta, float, int, None] = None,
    ) -> None:
        self.period = Period(steps=steps, samples=samples, time=time)
        self.last_samples: int = 0
        self.last_time: float = datetime.now().timestamp()

    def __call__(self, elapsed: Elapsed) -> bool:

        if self.period.steps is not None:
            return elapsed.steps % self.period.steps == 0

        if self.period.samples is not None:
            if elapsed.samples - self.last_samples >= self.period.samples:
                self.last_samples = elapsed.samples
                return True

        if self.period.time is not None:
            if elapsed.date - self.last_time >= self.period.time:
                self.last_time = elapsed.date
                return True

        return False


# ---------------------------------------
# callbacks
# ---------------------------------------


class tqdm_bar:
    def __init__(
        self,
        total: Union[Period, int, None] = None,
        desc=None,
        leave=True,
        file=None,
        ncols=None,
        mininterval=0.1,
        maxinterval=10.0,
        miniters=None,
        ascii=None,
        disable=False,
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format=None,
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        write_bytes=None,
        lock_args=None,
        nrows=None,
        colour=None,
        delay=0,
        gui=False,
        **kwargs,
    ):

        if isinstance(total, int):
            total = Period(steps=total)

        if total is not None:
            if total.steps is not None:
                bar_total = total.steps
                unit = "steps"
            elif total.samples is not None:
                bar_total = total.samples
                unit = "samples"
            elif total.time is not None:
                bar_total = total.time
                unit = "s"
                unit_scale = True
            elif total.date is not None:
                total.time = total.date - datetime.now().timestamp()
                bar_total = total.time
                unit = "s"
                unit_scale = True
            else:
                raise ValueError("Invalid total")
        else:
            bar_total = None
            unit = "it"

        self.total = total
        self.prev_step: Optional[int] = None
        self.prev_samples: Optional[int] = None
        self.prev_time: Optional[float] = None
        self.bar_total = bar_total
        self.bar = tqdm(
            desc=desc,
            total=bar_total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            delay=delay,
            gui=gui,
            **kwargs,
        )

    def __call__(self, state, batch, elapsed: Elapsed, loop):

        if self.total is None or self.total.steps is not None:
            if self.prev_step is None:
                self.prev_step = elapsed.steps - 1
            self.bar.update(elapsed.steps - self.prev_step)
            self.prev_step = elapsed.steps
        elif self.total.samples is not None:
            if self.prev_samples is None:
                self.prev_samples = elapsed.samples - get_batch_size(batch)
            self.bar.update(elapsed.samples - self.prev_samples)
            self.prev_samples = elapsed.samples
        elif self.total.time is not None:
            if self.prev_time is None:
                self.prev_time = elapsed._date_start
            self.bar.update(elapsed.date - self.prev_time)
            self.prev_time = elapsed.date
        else:
            raise ValueError("Invalid total")


def keras_bar(
    total: int,
    epoch=None,
    num_epochs=None,
    width=30,
    verbose=1,
    interval=0.05,
    stateful_metrics=None,
    always_stateful=False,
    unit_name="step",
):
    bar = Kbar(
        total,
        epoch=epoch,
        num_epochs=num_epochs,
        width=width,
        verbose=verbose,
        interval=interval,
        stateful_metrics=stateful_metrics,
        always_stateful=always_stateful,
        unit_name=unit_name,
    )

    def callback(state, batch, step, loop: Loop):
        assert loop.logs is not None
        bar.update(step, values=[(k, v) for k, v in loop.logs.items() if k != "step"])

    return callback
