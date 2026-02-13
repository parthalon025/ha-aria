# Dashboard - HA Intelligence Hub

Interactive web UI for visualizing hub data and managing automations.

## Overview

The dashboard provides a real-time view of:
- **Hub Status**: Module health, cache categories, recent events
- **Discovery**: Browse entities, devices, and areas
- **Capabilities**: View detected capabilities with entity counts
- **Predictions**: ML predictions with confidence scores
- **Patterns**: Detected behavioral patterns with LLM descriptions
- **Automations**: Review and approve/reject automation suggestions

## Technology Stack

- **Backend**: FastAPI (Jinja2 templates, WebSocket)
- **Frontend**: htmx (dynamic loading), Alpine.js (interactivity)
- **Styling**: Custom CSS (mobile-responsive)

## Structure

```
dashboard/
├── __init__.py         # Package init
├── routes.py           # FastAPI router with all endpoints
├── templates/          # Jinja2 HTML templates
│   ├── base.html       # Base layout with navigation
│   ├── home.html       # Hub status dashboard
│   ├── discovery.html  # Entity/device browser
│   ├── capabilities.html # Capability list
│   ├── predictions.html  # ML predictions
│   ├── patterns.html     # Detected patterns
│   └── automations.html  # Automation suggestions
└── static/             # Static assets
    ├── style.css       # Dashboard styles
    └── app.js          # Alpine.js helpers
```

## Access

- **URL**: http://localhost:8000/ui/
- **WebSocket**: ws://localhost:8000/ws

## Features

### Real-Time Updates

- WebSocket connection status indicator (top right)
- Auto-refresh on cache updates (no page reload needed)
- Live event stream on home page

### Entity Discovery

- Searchable entity list
- Filter by domain (light, sensor, switch, etc.)
- Shows entity state with color coding:
  - Green: on/home
  - Gray: off/away
  - Red: unavailable/unknown

### Capabilities

- Card-based grid layout
- Expandable entity lists
- Detection criteria display

### Predictions

- Confidence badges (high/medium/low)
- Current vs predicted values
- Model information
- Feature data (expandable)

### Patterns

- LLM-generated descriptions
- Pattern metadata (type, frequency, time window)
- Entity associations
- Raw data (expandable)

### Automations

- Approve/Reject buttons
- Confidence bars
- Trigger/Action/Condition display
- YAML export
- Status tracking (pending/approved/rejected)

## Implementation Notes

### WebSocket Integration

The base template sets up WebSocket connection on page load:

```javascript
// Connects to ws://localhost:8000/ws
// Listens for cache_updated events
// Updates relevant page sections via htmx
// Shows connection status indicator
```

### Cache Data Flow

1. Module updates cache → hub publishes `cache_updated` event
2. API broadcasts to all WebSocket clients
3. JavaScript handler triggers htmx reload for affected sections
4. Page sections re-fetch from cache via API

### Mobile-Responsive

- Flexbox/Grid layouts
- Collapsible navigation
- Touch-friendly controls
- Single-column on narrow screens

## Testing

All pages render correctly with test data:
- Home page: Shows hub health, modules, events
- Discovery: Displays entities with search/filter
- Capabilities: Card grid with expandable lists
- Predictions: ML predictions with confidence
- Patterns: Pattern cards with descriptions
- Automations: Suggestion cards with approve/reject

WebSocket functionality verified:
- Connection/reconnection works
- Cache updates broadcast correctly
- Ping/pong keepalive works

## Future Enhancements

- Real-time charts (predictions over time)
- Entity detail modals
- Automation editing UI
- Pattern visualization (timeline, heatmap)
- Export/import automation configs
- Dark mode toggle
