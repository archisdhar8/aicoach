import argparse

from app.api.dependencies import get_nba_data_repository, get_nba_data_service


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Play Lab data cache commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preload = subparsers.add_parser("preload-nba", help="Cache NBA teams, players, and rosters")
    preload.add_argument("--season", default="2025-26")
    arguments = parser.parse_args()

    get_nba_data_repository().initialize()
    if arguments.command == "preload-nba":
        directory_count, roster_count = get_nba_data_service().preload(arguments.season)
        print(
            f"Cached {directory_count} directory players and "
            f"{roster_count} roster entries for {arguments.season}."
        )


if __name__ == "__main__":
    main()
