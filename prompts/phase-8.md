Build a cinematic real-time dashboard for CrisisCoreEnv as a single self-contained HTML file called `dashboard.html`. It connects to the running FastAPI app and visualizes the agent in action.

The dashboard has 4 panels:

1. FLOOR MAP PANEL (left, 60% width)
   — Draw the building as a grid of zone rectangles. Each zone shows: zone ID label, occupancy count badge, color coding (green = safe, orange = hazard nearby, red = active hazard, gray = evacuated+empty, blue = exit zone)
   — People shown as small moving dots within their zone, repositioning when routes change
   — Stairwells between floors shown as connecting lines
   — Hazard zones pulse with a red border animation
   — Blocked exits shown with an X overlay
   — Floor selector tabs at the top if multi-floor

2. AGENT DECISION LOG PANEL (right top, 40% width)
   — Live scrolling feed of agent actions this episode
   — Each entry shows: tick number, which sub-agent acted (Evacuation / Dispatch / Comms), the action taken, and the reward received
   — Color code entries: green for positive reward, red for negative, gray for neutral
   — Auto-scroll to latest entry

3. METRICS PANEL (right middle)
   — 4 metric cards in a 2x2 grid: People Evacuated / Total, Current Tick / Max, Severity Level (badge with color), Episode Reward (running total)
   — All update in real time

4. RESPONDER PAYLOAD PANEL (right bottom)
   — Shows the most recent structured JSON payload sent to emergency services
   — Syntax-highlighted JSON display (use a monospace font, key in one color, value in another)
   — Shows: incident_type, location_coordinates (zone IDs), affected_persons count, hazards_present list, access_routes list
   — Animates in when a new dispatch action fires

CONTROLS at the bottom:
   — Start / Pause / Reset buttons
   — Speed slider (0.5x to 3x simulation speed)
   — Toggle: "Trained agent" vs "Random agent" (calls different endpoints or passes a flag)
   — "Trigger Crisis" button — fires a new hazard in a random zone mid-episode

IMPLEMENTATION:
   — Pure HTML/CSS/JS, no frameworks
   — Poll GET /state every 500ms when running
   — Use CSS grid for layout
   — Use requestAnimationFrame for smooth dot movement
   — Dark theme: background #0a0a0f, panels #12121a, accent colors per status
   — All colors and transitions should feel cinematic and urgent, not corporate
   — Add a subtle pulsing red dot in the top-left corner labeled "LIVE" when simulation is running