from __future__ import annotations

# Internal
import pickle
import typing as T
from copy import deepcopy
from base64 import b64decode, b64encode
from contextlib import suppress

# External
import click
from rich.live import Live
from rich.panel import Panel
from rich.style import StyleType
from rich.table import Row, Table, Column
from rich.errors import NotRenderableError
from rich.layout import Layout
from rich.pretty import Pretty
from rich.console import RenderableType
from rich.progress import TaskID, Progress
from rich.protocol import is_renderable

from fake_news_sis_simulator import EventType, Simulator, TimelineType, TopologyType


class ReversedTable(Table):
    def add_row(
        self,
        *renderables: T.Optional[RenderableType],
        style: T.Optional[StyleType] = None,
        end_section: bool = False,
    ) -> None:
        def add_cell(column: Column, renderable: RenderableType) -> None:
            column._cells.insert(0, renderable)

        cell_renderables: T.List[T.Optional[RenderableType]] = list(renderables)

        columns = self.columns
        if len(cell_renderables) < len(columns):
            cell_renderables = [
                *cell_renderables,
                *[None] * (len(columns) - len(cell_renderables)),
            ]
        for index, renderable in enumerate(cell_renderables):
            if index == len(columns):
                column = Column(_index=index)
                for _ in self.rows:
                    add_cell(column, T.Text(""))
                self.columns.append(column)
            else:
                column = columns[index]
            if renderable is None:
                add_cell(column, "")
            elif is_renderable(renderable):
                add_cell(column, renderable)
            else:
                raise NotRenderableError(
                    f"unable to render {type(renderable).__name__}; a string or other renderable object is required"
                )
        self.rows.insert(0, Row(style=style, end_section=end_section))


@click.command()
@click.argument("timeline_spec", type=int, nargs=-1)
@click.option("--iterations", "-i", type=int, default=0)
@click.option("--seed", type=str, default=None)
@click.option("--fifo", "timeline_type", flag_value=TimelineType.FIFO, default=True)
@click.option("--random", "timeline_type", flag_value=TimelineType.RND)
@click.option("--clique", "topology_type", flag_value=TopologyType.Clique, default=True)
@click.option("--star", "topology_type", flag_value=TopologyType.Star)
@click.option("--fake-rate", "-f", type=(float, float), required=True)
@click.option("--genuine-rate", "-g", type=(float, float), required=True)
def main(
    timeline_spec: T.List[int],
    *,
    seed: T.Union[str, bytes],
    fake_rate: T.Tuple[float, float],
    iterations: int,
    genuine_rate: T.Tuple[float, float],
    timeline_type: TimelineType,
    topology_type: TopologyType,
) -> None:
    timeline_size = len(timeline_spec) - 1
    if timeline_size < 1:
        raise ValueError("Timeline have at least 2 values")

    timeline = []
    template = [deepcopy(EventType.Fake) for _ in range(timeline_size)]
    for pos, spec in enumerate(timeline_spec):
        timeline += [deepcopy(template) for _ in range(spec)]
        template[timeline_size - (pos + 1)] = EventType.Genuine

    simulator = Simulator(
        timeline,
        rng=seed if seed is None else pickle.loads(b64decode(seed)),
        timeline_type=timeline_type,
        topology_type=topology_type,
        fake_rate_heuristic=lambda _: 1,
        genuine_rate_heuristic=lambda _: 1,
        internal_fake_transmission_rate=fake_rate[0],
        external_fake_transmission_rate=fake_rate[1],
        internal_genuine_transmission_rate=genuine_rate[0],
        external_genuine_transmission_rate=genuine_rate[1],
    )

    # === GUI Setup === #
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", ratio=7),
        Layout(name="info", ratio=15),
        Layout(name="stats", ratio=10),
        Layout(name="footer", ratio=1),
    )

    layout["header"].update(Panel(simulator, title="Simulator"))

    infos_table: ReversedTable

    def setup_info_table() -> None:
        nonlocal infos_table
        infos_table = ReversedTable(
            Column("Time", style="dim"),
            "Type",
            "Origin",
            title="Events",
            expand=True,
            show_header=True,
            header_style="bold white",
        )
        layout["info"].update(infos_table)

    setup_info_table()

    stats_table: ReversedTable

    def setup_stats_table(
        *headers: T.Union[Column, str],
    ) -> None:
        nonlocal stats_table
        stats_table = ReversedTable(
            *headers,
            title="Timeline Distribution",
            expand=True,
            show_header=True,
        )
        layout["stats"].update(stats_table)

    setup_stats_table()

    if iterations == 0:
        progress: T.Union[str, Progress] = "Press CTRL+C to exit..."
        progress_task: T.Optional[TaskID] = None
    else:
        progress = Progress(expand=True)
        progress_task = progress.add_task("[white]Simulating...", total=iterations)
    # FIXME: Rich has wrong typing definition
    layout["footer"].update(T.cast(RenderableType, progress))

    with Live(
        layout, refresh_per_second=10, screen=True, redirect_stdout=False, redirect_stderr=False
    ):
        seed = b64encode(pickle.dumps(simulator.rng))
        with suppress(KeyboardInterrupt):
            for idx, (time, event, stats) in enumerate(simulator):
                if 0 < iterations < idx:
                    break

                if len(infos_table.rows) > 40:
                    setup_info_table()
                infos_table.add_row(str(time), event.type.name, event.origin.name)

                stats_columns = tuple(
                    "".join(key.name[0] for key in keys) for keys in stats.keys()
                )
                if len(stats_columns) != len(stats_table.columns) or len(stats_table.rows) > 40:
                    setup_stats_table(*stats_columns)

                stats_table.add_row(*(Pretty(value) for value in stats.values()))

                if isinstance(progress, Progress) and progress_task is not None:
                    progress.update(progress_task, advance=1)

            layout["footer"].update("Press any key to exit...")
            input()

    print("Seed:", seed.decode(encoding="utf8"))


if __name__ == "__main__":
    main()
