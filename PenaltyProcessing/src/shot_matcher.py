#!/usr/bin/env python3
"""Match penalty shots to fixture position files and extract ball trajectories.

Pipeline summary:
- Resolve ordered fixture from penalties.csv row (home_team, away_team).
- Load the corresponding *_2_phases_positions.csv and keep only group name == Ball rows.
- Build trajectory window using local timestamps.
- Apply distance-based plausibility corrections for start time.
- Export one row per penalty with serialized trajectory and derived metrics.
- Export a dedicated release point so the plotted marker can be reconstructed later.
- Export issues for unresolved or partially resolved rows.
"""

#!/usr/bin/env python3
"""CLI entrypoint for penalty trajectory extraction."""

from penalty_processing import main


if __name__ == "__main__":
    main()
import unicodedata
