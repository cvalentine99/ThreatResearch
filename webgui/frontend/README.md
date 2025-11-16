# PCAP Analyzer - Frontend

SvelteKit-based frontend for GPU-accelerated PCAP analysis.

## Features

- **File Upload**: Drag & drop PCAP upload with progress tracking
- **Real-time Updates**: SSE streaming for live parse status
- **Interactive Visualizations**:
  - Network topology graph (Cytoscape.js)
  - Protocol distribution charts (ECharts)
  - Session list with filtering
- **Dark Theme**: Modern, cybersecurity-focused UI with Tailwind CSS + DaisyUI

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Development Server

```bash
npm run dev
```

Access at: http://localhost:5173

### 3. Build for Production

```bash
npm run build
```

Output directory: `build/`

### 4. Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── routes/
│   │   ├── +layout.svelte          # Main layout
│   │   ├── +page.svelte            # Upload page
│   │   ├── jobs/
│   │   │   └── +page.svelte        # Jobs list
│   │   └── dashboard/[jobId]/
│   │       └── +page.svelte        # Dashboard view
│   ├── lib/
│   │   ├── api/
│   │   │   └── client.js           # API client
│   │   └── components/
│   │       ├── NetworkGraph.svelte # Cytoscape topology
│   │       ├── StatsCharts.svelte  # ECharts visualizations
│   │       └── SessionList.svelte  # Session table
│   ├── app.css                      # Global styles
│   └── app.html                     # HTML template
├── package.json
├── svelte.config.js
├── vite.config.js
└── tailwind.config.js
```

## Technology Stack

- **Framework**: SvelteKit 2.0
- **Styling**: Tailwind CSS + DaisyUI
- **Charts**: Apache ECharts
- **Network Graph**: Cytoscape.js
- **Timeline**: Vis.js Timeline
- **Build Tool**: Vite

## API Integration

The frontend communicates with the FastAPI backend via:

- **REST API**: Upload, query packets/flows, statistics
- **SSE**: Real-time job status updates

API client: `src/lib/api/client.js`

## Development

### Type Checking

```bash
npm run check
```

### Linting

```bash
npm run lint
```

## Deployment

See main README.md for full deployment instructions with Nginx.

Quick steps:
1. Build: `npm run build`
2. Copy `build/` to `/var/www/pcap-analyzer/frontend/`
3. Configure Nginx to serve static files
4. Access via Nginx reverse proxy
