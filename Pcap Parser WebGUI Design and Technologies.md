

# **A Comprehensive Architectural Blueprint for a "Next-Generation" PCAP Visualization Dashboard**

## **Executive Summary: The "Next-Generation" PCAP Dashboard Stack**

This report delivers an expert-level architectural blueprint for transforming a standalone pcap parser into a high-performance, visually rich, "next-generation" web application. The analysis addresses the full project lifecycle: system architecture, data flow modeling, specific technology stack recommendations, advanced visualization components, and a complete production deployment guide for Ubuntu 24\.

The primary objective is to create a GUI that is not only functional but also aesthetically modern, highly interactive, and capable of handling the large, high-velocity datasets associated with network packet analysis.

To achieve this, the following "Golden Path" stack is recommended. This portfolio of technologies represents a "best-of-breed" selection for 2025, prioritizing performance, developer experience, and visual fidelity over all else:

* **System Architecture:** A decoupled, service-oriented architecture. This design features a persistent parsing service (the existing pcap parser), a search-optimized database for session metadata, a high-speed data API, and a lightweight, static frontend.  
* **Backend API:** **FastAPI (Python)**. This choice leverages its asynchronous (ASGI) performance, which rivals Node.js, while maintaining a native Python ecosystem that provides direct synergy with the existing pcap parser.  
* **Frontend Framework:** **SvelteKit**. This is a compiler-based framework that moves the majority of framework overhead from the browser (runtime) to the build step. This results in unparalleled runtime performance and minimal bundle sizes, which are critical for a data-heavy dashboard.  
* **Data Store:** **OpenSearch** (or its progenitor, Elasticsearch). This is the industry-standard solution for indexing, searching, and aggregating the vast amounts of network session metadata (known as SPI data) that the pcap parser will generate.  
* **Real-Time Transport:** **Server-Sent Events (SSE)**. For live monitoring, SSE provides a simple, robust, and highly efficient unidirectional data stream from the server to the client, complete with automatic reconnection and native browser support.  
* **Visualization Portfolio:** No single library can provide "next-gen" visuals for all pcap data types. A specialized portfolio is required:  
  * **General Charting:** **Apache ECharts** for its high-performance rendering of millions of data points and its vast library of chart types (lines, pies, heatmaps, maps).  
  * **Network Topology:** **Cytoscape.js** for its powerful, analysis-focused graph theory engine, designed specifically for complex network visualization.  
  * **Timelines:** **Vis.js (Timeline)** for its dedicated, interactive timeline component, ideal for recreating Wireshark-style packet flow diagrams.  
* **Deployment:** **Nginx** will serve as the production-grade reverse proxy. It will serve the static SvelteKit frontend build and route all API and streaming requests to the **Gunicorn/Uvicorn** backend service, all running on the **Ubuntu 24** server.

---

## **I. Architecting a "Next-Generation" PCAP Visualizer**

The foundation of a "next-gen" tool is not its visual flair, but its architecture. The core challenge of pcap analysis is handling a high-throughput data-write (parsing) operation while simultaneously serving a low-latency data-read (visualization) operation. A monolithic design, where the web server parses pcap files on-demand, is destined for failure; it will lock up, time out, and provide a non-responsive user experience.

### **A. The Core Principle: A Decoupled, Service-Oriented Architecture**

The primary architectural principle for this system must be *decoupling*.1 A decoupled architecture, where each software component (parser, database, web portal) is deployed as a dedicated service, is the standard for building scalable, high-performance systems. This approach allows each component to be optimized and scaled independently, preventing a bottleneck in one service (like a large parsing job) from crippling the others (like the interactive UI).1

This model is validated by industry-standard open-source tools, most notably **Arkime** (formerly Moloch).2 The Arkime system is composed of three distinct components, which provides a perfect model for this project:

1. **capture:** A high-performance, threaded C application that monitors network traffic, parses packets, and sends metadata (known as SPI data, or Session Profile Information) to the database.2 This is the production-grade role that the existing pcap parser will evolve to fill.  
2. **OpenSearch/Elasticsearch:** The search database technology that powers the system.2 This is the central data store for all parsed metadata.  
3. **viewer:** A Node.js application that handles the web interface and API queries, allowing users to browse and search the data.2 This is the web GUI and API layer that will be built.

Adopting this architecture effectively separates the *write-intensive* parsing (a continuous, high-throughput, background process) from the *read-intensive* visualization (an on-demand, bursty, user-facing query process).

### **B. Data Flow Model 1: "Static Analysis" (File-Upload Batch Processing)**

This is the most logical first implementation, supporting the analysis of pre-existing pcap files.4 It is not for live traffic monitoring but is essential for forensic analysis.

The data flow is asynchronous to prevent API timeouts:

1. \*\*\*\* The user accesses the web GUI and is presented with an upload interface. They select a .pcap or .pcapng file from their local disk. Tools like Arkime confirm the feasibility of a UI-based upload feature.5  
2. \*\*\*\* Nginx receives the (potentially large) file upload and routes it to the backend API.  
3. \*\*\*\* The file is received at a secure endpoint (e.g., /api/upload). The API *does not* begin parsing the file. Doing so would block the server thread and lead to an HTTP timeout. Instead, it saves the file to a temporary, persistent volume.  
4. \*\*\*\* The API endpoint's only job is to create a new "parse" task in a dedicated task queue (e.g., Celery with a Redis or RabbitMQ broker). The job message contains the path to the uploaded pcap file.  
5. \*\*\*\* A separate, long-running worker process (running the pcap parser, which may use libraries like Scapy 7 or shell out to tshark 8) polls the task queue. It picks up the job, reads the pcap file, extracts all relevant metadata (flows, protocols, IPs, timestamps), and inserts this metadata as structured JSON documents into the **OpenSearch** database.  
6. \*\*\*\* While the worker parses, the UI polls a "job status" endpoint (e.g., /api/upload/status/\<job\_id\>). Once the API confirms the job is complete, the client application is "unlocked," allowing the user to navigate to the dashboard views.  
7. **\[Visualization\]** The client's dashboard views now make API calls (e.g., /api/sessions) to query the OpenSearch database, which instantly returns the newly parsed data to populate the visualizations.

This asynchronous, batch-processing model is robust and addresses the performance and timing challenges inherent in pcap analysis, where a large file could take minutes to process.7

### **C. Data Flow Model 2: "Real-Time" (Live Capture and Streaming)**

This is the true "next-generation" architecture, designed for a live dashboard that monitors a network interface in real-time.9

The data flow is a continuous stream:

1. \*\*\*\* The pcap parser is refactored from a simple script into a long-running, persistent service (e.g., a systemd daemon on the Ubuntu server). It directly uses a library like libpcap 11 (via a Python wrapper like pyshark or pcapy 8) to sniff a specified network interface (e.g., eth0) in real-time.  
2. \*\*\*\* As the parser identifies packets, it performs session analysis (similar to Zeek 12), aggregates them into flows, and immediately serializes them into structured JSON metadata.  
3. \*\*\*\* This JSON metadata is streamed, in real-time, directly into an OpenSearch index.14 This index serves as the "hot" data store for all recent traffic.  
4. \*\*\*\* The FastAPI server is now largely stateless. It does not perform any parsing; its role is to query OpenSearch and stream data to connected clients.  
5. \*\*\*\* When a user opens the "Live View" on the dashboard, the SvelteKit frontend establishes a **Server-Sent Event (SSE)** connection to a specific endpoint on the FastAPI backend, such as /api/stream.15  
6. \*\*\*\* The FastAPI endpoint, using its native StreamingResponse, continuously monitors the OpenSearch database for new entries (e.g., timestamp \> last\_seen\_timestamp) or, for a more advanced setup, listens to a message queue like Apache Kafka 10 that the parser writes to. As new session data arrives, it is pushed down the SSE connection to the client.  
7. \*\*\*\* The Svelte frontend's EventSource listener receives these new JSON data events and dynamically updates the visualizations (e.g., adding new nodes to the Cytoscape graph, appending data to the ECharts time-series) without requiring a single page refresh.17

### **D. Architectural Implications**

A review of these two models reveals two critical considerations that dictate the project's success.

First, the pcap parser itself is the primary performance bottleneck, and its underlying technology dictates the system's capabilities. If the parser is a pure Python script (e.g., using Scapy 7), its performance will be insufficient for high-speed, real-time capture. Python's overhead will cause packet drops. Therefore, a Python-based parser *must* be limited to the "Static Analysis" model (Model 1\) or live-monitoring of very low-traffic interfaces. To successfully implement the "Real-Time" model (Model 2\) for a production network, the parser must be re-architected as a high-performance C/C++ or Rust application (like Arkime's capture 2) or integrate an external, compiled tool like Zeek 12 or tshark 8 as the core parsing engine.

Second, the data store is the key to a "rich" and "fast" UI, not the API. A "rich in visuals" GUI implies the need to run complex, multi-faceted queries on the fly (e.g., "group all traffic by protocol," "show a timeline of flows between host A and host B," "find all sessions with a specific HTTP user-agent").19 Attempting to run these queries against flat files or a traditional SQL database will be unacceptably slow. The research and industry precedent overwhelmingly point to **OpenSearch** (or Elasticsearch) as the correct database for network metadata.2 Arkime refers to this as "Session Profile Information" (SPI) data.13

Therefore, the pcap parser's most important job is not just parsing, but *transforming* raw packets into structured JSON documents (sessions) that can be indexed in OpenSearch. The "speed" of the GUI will be a direct function of the *query speed* of OpenSearch. The FastAPI backend becomes a thin, simple, and fast proxy that translates user requests into OpenSearch aggregation queries.

---

## **II. The "Golden Path" Stack: Technology Recommendations for 2025**

This section provides a prescriptive justification for each layer of the application stack, aligning with the "next-gen" and high-performance requirements.

### **A. Backend API: FastAPI (Python)**

While Node.js is a strong, conventional choice for real-time applications due to its event-driven, non-blocking I/O 22, **FastAPI** is the superior and more logical choice for this specific project.

1. **Python-Native Synergy:** The project's core—the pcap parser—is already written in Python, which is the dominant language for data analysis and security tools (e.g., Scapy, PcapXray).7 Choosing FastAPI creates a single-language ecosystem.24 This dramatically simplifies development, dependency management (e.g., sharing a single requirements.txt or pyproject.toml), and the ability to share data models (Pydantic classes) between the parser, the worker, and the API. This aligns with the modern "Python Stack" (Python backend, JS frontend).25  
2. **Unmatched Performance (in Python):** FastAPI is one of the fastest Python frameworks available, with performance benchmarked as on-par with Node.js for I/O-bound tasks.23 It is built on the Starlette ASGI framework and Pydantic 23, providing native async/await support from the ground up.27 This asynchronicity is essential for efficiently handling a high number of concurrent API requests (from multiple dashboard users) and managing real-time data streams (SSE/WebSockets).  
3. **Superior Developer Experience for APIs:** For a data-heavy, API-driven application, FastAPI's features are purpose-built to accelerate development and reduce errors:  
   * **Automatic Data Validation:** By using Pydantic models, FastAPI automatically validates, parses, and type-checks all incoming request data.26 This is a non-negotiable feature for handling complex and potentially malformed pcap metadata.  
   * **Automatic API Docs:** It auto-generates interactive OpenAPI (Swagger) and ReDoc documentation.26 This is invaluable for developing and debugging the frontend-backend connection, as the SvelteKit developer can test API endpoints in the browser.

In conclusion, FastAPI offers the raw async performance required for a real-time dashboard 28 while providing the rapid development, data-handling, and ecosystem benefits of Python.23

### **B. Frontend Framework: SvelteKit (Svelte)**

This is the most critical "next-gen" decision. The default, "safe" choice for many developers is React 31, which boasts a "massive" ecosystem.33 However, for a high-performance dashboard, **Svelte** is the architecturally superior choice.

1. **The Compiler Advantage (No Virtual DOM):** React and Vue are *runtime* libraries.31 They must ship a significant amount of framework code to the user's browser. This code then runs in the browser, "diffing" a Virtual DOM (VDOM) and calculating how to update the UI. Svelte is a *compiler*.34 It shifts this work to the *build step* (npm run build). It compiles .svelte components into tiny, highly optimized, "surgical" vanilla JavaScript that interacts with the DOM directly.34  
2. **Performance & Bundle Size:** The "next-gen" user experience is defeated by a five-second load time. Because Svelte is a compiler, it ships *no framework runtime*. This results in dramatically smaller bundle sizes. A "Hello World" app in Svelte is approximately 1.6kb (gzipped), compared to \~40kb for React (plus its own libraries).36 For a data-heavy dashboard, this means a faster initial load (Time to Interactive) and superior runtime performance, as updates do not pay the "tax" of a VDOM diff.35  
3. **Superior Developer Experience:** Svelte's syntax is simpler, requires less boilerplate, and is more intuitive. Reactivity is built into the language (e.g., let count \= 0), not bolted on with hooks (e.g., useState).36 Features that require third-party libraries in React, such as native state management (stores), animations, and transitions, are built-in 36, reducing bloat and complexity.  
4. **SvelteKit (The Meta-Framework):** SvelteKit is the official meta-framework for Svelte, analogous to Next.js for React.33 It provides a complete, integrated solution with file-based routing, server-side rendering (SSR), API routes, and static site generation, making it a production-ready tool.39

The only significant trade-off is that Svelte's ecosystem (third-party components, libraries) is smaller and less mature than React's.33 This concern is mitigated by our visualization strategy, which relies on powerful, framework-agnostic JavaScript libraries (ECharts, Cytoscape.js, Vis.js) that integrate cleanly into any framework, including Svelte.

### **C. Real-Time Data Streaming: Server-Sent Events (SSE)**

For the real-time architecture (Model 2), the application must push live data to the client. The choice is between WebSockets and Server-Sent Events (SSE).

1. **Unidirectional vs. Bidirectional:** The fundamental difference is the direction of data flow. A monitoring dashboard is a *unidirectional* (one-way) stream: the server *pushes* data to the client, and the client simply listens.43 WebSockets provide a *bidirectional*, full-duplex communication channel.44 They are designed for applications like chat, real-time collaboration, and multiplayer gaming, where the client must also send a high-frequency stream of data to the server.45 Using WebSockets for a read-only dashboard is "a significant overkill" 44 and introduces unnecessary complexity.  
2. **Simplicity & Robustness:** SSE is built entirely on standard HTTP.44 It requires no special protocol handshake and is far more likely to work through corporate firewalls and packet-inspecting proxies.43 Critically, the browser's native EventSource API handles **automatic reconnection** if the connection drops.43 With WebSockets, this is a feature that must be manually implemented on the client.48  
3. **Native Framework Support:** FastAPI has excellent, native support for SSE via its StreamingResponse object, making implementation trivial.16

For a dashboard use case, **SSE** is the technically correct, simpler, and more robust choice for streaming real-time monitoring data.45

### **D. Tables for Analysis and Decision-Making**

To summarize the key technology selections, the following tables provide a direct comparison of the leading options.

**Table 1: Backend Framework Comparison (FastAPI vs. Node.js)**

| Feature | FastAPI (Python) | Node.js (JavaScript/TypeScript) |
| :---- | :---- | :---- |
| **Core Language** | Python | JavaScript / TypeScript |
| **Performance (Async Model)** | High (ASGI). On par with Node.js for I/O-bound tasks.23 | High (Event-driven, non-blocking I/O).22 |
| **Ecosystem Synergy** | **Excellent.** Natively integrates with Python parser (Scapy, etc.).24 | Poor. Requires cross-language communication or parser rewrite. |
| **Development Speed** | **Very Fast.** Minimal boilerplate.29 | Fast, but can be more complex.29 |
| **Data Validation** | **Automatic.** Built-in with Pydantic.26 | Manual. Requires third-party libraries (e.g., Zod, Joi). |
| **API Auto-Documentation** | **Automatic.** Built-in (Swagger/OpenAPI).26 | Manual. Requires third-party libraries (e.g., Swagger-JSDoc). |
| **Real-time Suitability** | Excellent. Native async, SSE, and WebSocket support.16 | Excellent. Ideal for real-time, concurrent apps.23 |

**Table 2: Frontend Framework Showdown (SvelteKit vs. Next.js/React)**

| Feature | SvelteKit (Svelte) | Next.js (React) |
| :---- | :---- | :---- |
| **Core Architecture** | **Compiler.** No framework runtime in browser.36 | **Runtime.** Virtual DOM diffing in browser.36 |
| **Performance Model** | **Excellent.** Surgical DOM updates.33 | Good. VDOM reconciliation is an overhead.33 |
| **Avg. Bundle Size** | **Minimal** (e.g., \~1.6kb "Hello World").36 | **Large** (e.g., \~40kb+ "Hello World").36 |
| **Built-in Features** | **Rich.** Native state mgt., animations, transitions.36 | **Minimal.** Relies on third-party libraries for state, etc. |
| **Ecosystem Maturity** | Moderate. Growing, but smaller.33 | **Massive.** Largest ecosystem.33 |
| **Developer Experience** | **Excellent.** Simple, intuitive, less boilerplate.36 | Good, but complex (Hooks, VDOM rules).36 |

**Table 3: Real-Time Protocol Selection (SSE vs. WebSockets)**

| Feature | Server-Sent Events (SSE) | WebSockets |
| :---- | :---- | :---- |
| **Communication Direction** | **Unidirectional** (Server-to-Client).44 | **Bidirectional** (Full-Duplex).44 |
| **Underlying Protocol** | Standard **HTTP/HTTPS**.44 | Custom ws:// or wss:// protocol (upgraded from HTTP).44 |
| **Automatic Reconnection** | **Yes.** Built into the native EventSource API.43 | **No.** Must be implemented manually in JavaScript.48 |
| **Binary Data Support** | No. Text-only (UTF-8).44 | **Yes.** Supports text and binary frames.44 |
| **Typical Use Case** | **Dashboards, notifications, live feeds**.45 | **Chat apps, multiplayer games, live collaboration**.45 |
| **Implementation Complexity** | **Low.** Simple server endpoint, native browser client.51 | Moderate. Requires handshake and connection management.55 |

---

## **III. Designing the "Rich Visuals": A Component-by-Component Guide**

This section translates the "next-gen" and "rich visuals" requirement into a concrete visual and technical specification. The dashboard will be designed by breaking it down into its core functional components and mapping each one to the best-in-class visualization library for that specific task.

### **A. The "Next-Gen" Aesthetic: UI/UX Principles & Toolkit**

The "next-gen" aesthetic for network monitoring and cybersecurity dashboards is defined by a clear move *away* from cluttered, table-based layouts and *towards* clean, high-information-density, dark-mode-first interfaces.58

* **Design Targets:** The visual identity will draw inspiration from:  
  * **Dribbble:** The "cybersecurity dashboard" 65 and "network monitoring" 58 tags show a consistent trend: dark backgrounds (deep blue, dark grey, or black), with bright, neon-like accent colors for data visualization, and modular, card-based layouts.67  
  * **Cisco Meraki:** The new Meraki dashboard is a prime example of modern enterprise UI. It features a sleek "midnight blue" header, clean navigation, and a focus on simplifying complex network data into digestible visuals.68  
  * **Arkime (Moloch):** The open-source standard provides a *functional* benchmark. Its UI, while dated, demonstrates the *essential components* that a pcap visualizer must have: a "Sessions" list, an "SPI" (Session Profile Information) view, a temporal graph ("SPI Graph"), and a network graph ("Connections").20 The goal is to build a modern, aesthetically superior version of these exact components.  
* **UI Toolkit Recommendation:**  
  * **CSS Framework: Tailwind CSS.** To achieve a custom, "next-gen" design, it is best to avoid opinionated component libraries like Material-UI 71 or Ant Design 72, which can make applications look generic. Tailwind is a utility-first CSS framework that provides complete control over the visual layer, enabling the creation of a unique, branded-feeling dashboard.  
  * **Component Library (Optional): daisyUI.** This is a plugin for Tailwind CSS that provides pre-built, unstyled component class names (e.g., btn, modal).73 It offers the convenience of a component library without sacrificing the customizability of Tailwind, making it an ideal partner for Svelte.

### **B. The Visualization Portfolio: Using the Right Tool for Each Job**

A "portfolio" of visualization libraries is required because there is no single "best" library for all data types. A common mistake is to choose one library and force it to perform tasks it was not designed for.

* **D3.js** 75 is the "big dog" of visualization. It is a low-level *engine* for *creating* novel, custom visualizations by binding data to DOM elements.78 It provides unparalleled flexibility but has a notoriously steep learning curve 76 and is verbose for creating standard charts.  
* **Apache ECharts** 75 is a high-level *library* for *charts*. It offers 90% of the beauty of D3 for 10% of the effort and, most importantly, has *vastly* superior performance for large datasets.80  
* **Cytoscape.js** 83 is a specialized *graph theory* library, not just a visualizer. It is purpose-built for network analysis.83  
* **Vis.js** 86 is a collection of components, including a specialized *timeline* library.86

The "next-gen" architect's approach is to assemble a portfolio of these best-in-class, framework-agnostic libraries. All are pure JavaScript and will integrate cleanly into Svelte.  
The Portfolio: ECharts (charts/maps), Cytoscape.js (topology), and Vis.js (timelines).

### **C. Core Visualization Component 1: The Interactive Network Topology Graph**

* **Goal:** To visually map the connections between nodes (IP addresses, devices) in the pcap data. This is the "Connections" view in Arkime 20 or the dynamic topology map in a Cisco Meraki dashboard.88  
* **Recommended Library:** **Cytoscape.js**.83  
* **Rationale:** While other libraries like Sigma.js 89 (focused on high-speed rendering) and Vis.js Network 86 (a general-purpose network visualizer) exist, Cytoscape.js is the clear winner for this use case. It is not just a "visualizer"; it is a "graph theory... library for analysis and visualisation".83 It is designed to handle large, complex networks 84 and includes graph theory algorithms (e.g., shortest path, node degree, hub detection) out of the box.83 This allows the UI to not only *show* the graph but *analyze* it.  
* **Implementation:**  
  1. The FastAPI backend will provide an endpoint (e.g., /api/graph) that queries OpenSearch for all sessions matching the user's filter.  
  2. The API will process this list of sessions into a Cytoscape-compatible JSON format: an array of nodes (e.g., { data: { id: '1.2.3.4', type: 'internal' } }) and an array of edges (e.g., { data: { source: '1.2.3.4', target: '8.8.8.8', protocol: 'DNS' } }).  
  3. The Svelte component will fetch this JSON and instantiate a Cytoscape.js instance, applying a "force-directed" layout (like d3-force 92) to automatically position the nodes in a readable, organic cluster.

### **D. Core Visualization Component 2: The Packet Flow Timeline**

* **Goal:** To replicate the functionality of Wireshark's "Flow Graph".94 This visualization is a sequential, vertical timeline that shows the back-and-forth conversation (e.g., TCP SYN, SYN-ACK, ACK) between two specific hosts, allowing for deep analysis of a single connection.  
* **Recommended Library:** **Vis.js (Timeline Component)**.86  
* **Rationale:** The Vis.js Timeline component is *purpose-built* for this exact task.86 It creates an interactive, zoomable timeline that can render items as either a single point in time (a single packet) or a range (a connection duration).87 It also supports "groups" to create vertical lanes. This is a far more robust and feature-complete solution than attempting to build a custom Gantt-like chart from scratch.97  
* **Implementation:**  
  1. When a user clicks an "edge" (a conversation) in the Cytoscape graph, the UI will capture the source and destination IPs and call a new API endpoint (e.g., /api/flow?ip\_a=1.2.3.4\&ip\_b=8.8.8.8).  
  2. The FastAPI backend will query OpenSearch for all individual packets associated with that session, sorted by timestamp.  
  3. It will format this packet list into a Vis.js DataSet 96, where each packet is an item. The data will be grouped into two lanes: "Host A" and "Host B."  
  4. A Svelte component (likely in a modal) will render this DataSet using vis-timeline, instantly creating a "Wireshark-style" 94 flow diagram.

### **E. Core Visualization Component 3: Statistical Dashboards (Bandwidth, Protocols)**

* **Goal:** To provide high-level summaries of the entire dataset, such as "Bandwidth Over Time" 98 and "Protocol Distribution".100  
* **Recommended Library:** **Apache ECharts**.80  
* **Rationale:** D3.js is too low-level and slow to develop for standard charts.78 ECharts is the "next-gen" choice because it is explicitly designed for *high performance* and *large datasets*.80 Its documented ability to render "10 million data in realtime" 80 using progressive rendering is precisely what a pcap visualizer needs. It also has a massive gallery of "next-gen" chart types built-in, including Line, Bar, Pie, Sankey, Chord, and Treemaps.80  
* **Implementation:**  
  1. **Bandwidth Over Time:** This will use an ECharts "Large scale area chart".104 The API will perform an OpenSearch "date histogram" aggregation to bucket traffic volume (sum of bytes) into time intervals (e.g., per second, per minute). ECharts will render this time-series data flawlessly.99  
  2. **Protocol Distribution:** This will use an ECharts "Donut Chart" 101 or "Treemap" 101 (which is better for hierarchical data). The API will perform an OpenSearch "terms" aggregation to get the count of sessions for each protocol (e.To, UDP, ICMP, etc.).

### **F. Core Visualization Component 4: The Conversation Matrix (Heatmap)**

* **Goal:** To visualize the *volume* of traffic between many different hosts simultaneously. A network graph (Component 1\) becomes an unreadable "hairball" when thousands of nodes are displayed.109 A matrix heatmap is the "next-gen" solution to this problem.  
* **Recommended Library:** **ECharts (Heatmap Component)**.80  
* **Rationale:** This visualization maps directly to a "Heat Map".110 The X-axis represents Source IPs, the Y-axis represents Destination IPs, and the color of the cell (from cool to hot) represents the volume of traffic (total bytes or packet count). This is a powerful, high-density way to visualize an "adjacency matrix".109 ECharts has a built-in, high-performance heatmap chart 80 ideal for this.  
* **Implementation:** The API will perform an OpenSearch "terms" aggregation on *both* source.ip and destination.ip simultaneously (a 2D aggregation) to create the matrix data.109 ECharts will then render this 2D array as a heatmap.

### **G. Core Visualization Component 5: Geo-IP Mapping**

* **Goal:** To display the geographic location of source and destination IPs on a world map, highlighting international traffic. This is a key feature of existing tools like CapAnalysis.114  
* **Recommended Library:** **ECharts (GEO/Map Component)**.80  
* **Rationale:** ECharts provides extensive and beautiful geospatial visualization capabilities out of the box.104 It includes components for plotting points on maps, as well as "lines series for directional information" 80, as seen in its "GEO SVG Traffic," "Flights," and "Flights GL" examples.104 This is far more integrated and visually impressive than trying to wire D3 to a separate library like Leaflet.  
* **Implementation:** The backend API will take the list of external IPs from an OpenSearch query, look them up using an integrated Geo-IP database (like the one CapAnalysis uses 114), and return a list of coordinates and connection pairs. ECharts will then render these as glowing, arcing lines on a 2D or 3D globe.

### **H. Table for Analysis and Decision-Making**

**Table 4: Recommended Visualization Library by Use Case**

| Visual Goal | Inspired By | Recommended Library | Rationale / Key Features |
| :---- | :---- | :---- | :---- |
| **Interactive Network Topology** | Arkime "Connections" 20, Meraki Topology 88 | **Cytoscape.js** 83 | Purpose-built for graph theory & analysis, not just visuals. Handles large, complex networks.84 |
| **Sequential Packet Flow Timeline** | Wireshark "Flow Graph" 94 | **Vis.js (Timeline)** 86 | Dedicated, interactive timeline component. Supports items, ranges, and groups. Zoomable/pannable.87 |
| **Statistical Charts** (Bandwidth, Protocols) | Standard Dashboards 99 | **Apache ECharts** 80 | **High-performance:** Renders millions of data points.80 Massive chart gallery.104 |
| **Conversation Volume Matrix** | Adjacency Matrix Plots 109 | **ECharts (Heatmap)** 80 | High-density, "next-gen" alternative to cluttered graphs. Represents 2D matrix data perfectly.109 |
| **Geospatial IP Map** | CapAnalysis "GeoMAP" 114 | **ECharts (GEO/Map)** 80 | Built-in 2D/3D maps, line/flow visualization ("Flights GL"), and global plotting.104 |

---

## **IV. Production Deployment on Ubuntu 24**

This section provides a complete, step-by-step guide for deploying the **SvelteKit \+ FastAPI** stack on a production Ubuntu 24 server, using Nginx as a reverse proxy and systemd for service management.

### **A. Part 1: Server Preparation & Firewall Configuration (Ubuntu 24\)**

1. **Update System:** Log into the Ubuntu 24 server and ensure all packages are up to date.  
   Bash  
   sudo apt update && sudo apt upgrade \-y 

2. **Install Core Dependencies:** Install Python 3, the virtual environment module, pip, Nginx, and Git.  
   Bash  
   sudo apt install python3-pip python3-venv nginx git \-y \[117, 118, 119\]

3. **Install Node.js:** The SvelteKit frontend requires a Node.js environment to run its build process. The default Ubuntu apt repositories can have outdated versions.120 It is best practice to use the official NodeSource repository to install the latest Long-Term Support (LTS) version.  
   Bash  
   curl \-fsSL https://deb.nodesource.com/setup\_lts.x | sudo \-E bash \- \[121\]  
   sudo apt-get install \-y nodejs \[121\]

4. **Configure Firewall (UFW):** Configure the "Uncomplicated Firewall" to allow SSH (so the connection is not dropped) and Nginx (HTTP/HTTPS) traffic.  
   Bash  
   sudo ufw allow OpenSSH \[118, 122\]  
   sudo ufw allow 'Nginx Full'   
   sudo ufw enable

### **B. Part 2: Deploying the FastAPI Backend**

1. **Clone Project:** Clone the project repository into a production-ready directory, such as /opt.  
   Bash  
   sudo git clone https://your-repo-url.com/pcap-gui /opt/pcap-gui \[118, 119\]

2. **Create Virtual Environment:** Create a self-contained Python virtual environment for the backend.  
   Bash  
   sudo python3 \-m venv /opt/pcap-gui/backend/venv \[117, 119, 123\]

3. **Install Dependencies:** Activate the environment and install the required Python packages from the requirements.txt file. This file *must* include fastapi, uvicorn, gunicorn, and opensearch-py (or elasticsearch).  
   Bash  
   source /opt/pcap-gui/backend/venv/bin/activate \[119\]  
   pip install \-r /opt/pcap-gui/backend/requirements.txt \[117, 119\]

4. **Create Systemd Service:** To ensure the FastAPI application runs as a persistent service (and restarts on failure or reboot), create a systemd service file.117  
   * Create the file: sudo vim /etc/systemd/system/fastapi.service  
   * Add the following configuration, adjusting User and paths as needed 118:  
     Ini, TOML  
     \[Unit\]  
     Description\=PCAP GUI FastAPI Backend Service  
     After\=network.target

     User\=your-non-root-user  
     WorkingDirectory\=/opt/pcap-gui/backend  
     Environment\="PATH=/opt/pcap-gui/backend/venv/bin"  
     ExecStart\=/opt/pcap-gui/backend/venv/bin/gunicorn \--workers 4 \\  
               \--worker-class uvicorn.workers.UvicornWorker \\  
               main:app \--bind 127.0.0.1:8000  
     Restart\=always

     \[Install\]  
     WantedBy\=multi-user.target

   * This command uses Gunicorn as a process manager to run 4 Uvicorn workers, creating a robust, multi-process service bound only to localhost.117  
5. **Enable Service:** Start and enable the new service.  
   Bash  
   sudo systemctl daemon-reload   
   sudo systemctl start fastapi   
   sudo systemctl enable fastapi 

   The backend is now running and listening on http://127.0.0.1:8000.

### **C. Part 3: Deploying the SvelteKit Frontend**

1. **Build the Application:** Navigate to the frontend directory and install dependencies.  
   Bash  
   cd /opt/pcap-gui/frontend  
   npm install \[122, 123\]

2. **Run Build:** Use the SvelteKit build command. This compiles the .svelte files into optimized, static HTML, CSS, and JavaScript.36  
   Bash  
   npm run build

   This process generates a build or dist directory containing the production-ready application.  
3. **Copy Files:** Create a directory for Nginx to serve these files and copy them into place.  
   Bash  
   sudo mkdir \-p /var/www/pcap-gui  
   sudo cp \-R /opt/pcap-gui/frontend/build /var/www/pcap-gui/

### **D. Part 4: Configuring the Nginx Reverse Proxy**

This is the final step that unifies the application. Nginx will act as the single public entry point, serving the static SvelteKit app and proxying dynamic requests to the FastAPI backend.118

1. **Create Nginx Configuration:** Create a new Nginx config file.  
   Bash  
   sudo vim /etc/nginx/sites-available/pcap-gui

2. **Add Configuration:** This configuration serves the SvelteKit app from /, proxies all API requests from /api to the FastAPI backend, and correctly handles the SSE streaming endpoint.  
   Nginx  
   server {  
       listen 80;  
       server\_name pcap.your-domain.com; \# Replace with your domain or IP

       \# 1\. Serve the SvelteKit Frontend  
       location / {  
           root /var/www/pcap-gui/build;  
           index index.html;  
           \# This line is CRITICAL for SvelteKit/React client-side routing  
           try\_files $uri $uri/ /index.html; \[125\]  
       }

       \# 2\. Proxy API requests to the FastAPI backend  
       location /api {  
           proxy\_pass http://127.0.0.1:8000; \[118, 126\]  
           proxy\_set\_header Host $host;  
           proxy\_set\_header X-Real-IP $remote\_addr;  
           proxy\_set\_header X-Forwarded-For $proxy\_add\_x\_forwarded\_for;  
           proxy\_set\_header X-Forwarded-Proto $scheme;  
       }

       \# 3\. Proxy Real-Time SSE Stream  
       \# This location requires special headers to disable buffering  
       location /api/stream { \# Match the specific streaming endpoint  
           proxy\_pass http://127.0.0.1:8000; \# Assuming stream is at /api/stream

           \# Headers for SSE  
           proxy\_http\_version 1.1;  
           proxy\_set\_header Connection '';  
           proxy\_set\_header Cache-Control no\-cache;  
           proxy\_set\_header Accept text/event-stream;

           \# This is the most critical directive for streaming  
           proxy\_buffering off;  
       }  
   }

3. **Enable Site and Reload Nginx:**  
   Bash  
   sudo ln \-s /etc/nginx/sites-available/pcap-gui /etc/nginx/sites-enabled/ \[118, 122\]  
   sudo nginx \-t \[118, 122\]  
   sudo systemctl restart nginx

### **E. Part 5: Securing with HTTPS (Let's Encrypt)**

Never run a production dashboard over unencrypted HTTP.

1. **Install Certbot:** This tool automates the process of obtaining and renewing SSL certificates.  
   Bash  
   sudo apt install certbot python3-certbot-nginx \-y \[125\]

2. **Run Certbot:** This command will automatically detect the server\_name from the Nginx config, obtain a certificate, and update the config file to handle SSL and redirect HTTP to HTTPS.  
   Bash  
   sudo certbot \--nginx \-d pcap.your-domain.com \[125\]

3. **Verify Auto-Renewal:** Certbot sets up an automatic renewal process.  
   Bash  
   sudo certbot renew \--dry-run \[117, 125\]

The application is now fully deployed, secured, and accessible at https://pcap.your-domain.com.

### **F. Deployment Implications**

This Nginx reverse proxy architecture is not just for convenience; it is a critical component for simplicity and security. A common failure point in modern decoupled applications is Cross-Origin Resource Sharing (CORS). If the frontend (at pcap.your-domain.com) tries to fetch data from the API (at localhost:8000), the browser will block the request as a security risk.124

The architecture outlined above *completely avoids this problem*. By having Nginx serve *both* the SvelteKit app (from /) and proxy the FastAPI app (from /api) from the *exact same domain*, all requests are considered "same-origin" by the browser. This eliminates the need for complex and fragile Access-Control-Allow-Origin headers in the FastAPI backend, resulting in a cleaner, more secure, and easier-to-manage production system.124

---

## **V. Conclusion: Your "Next-Gen" Roadmap**

This report has provided a comprehensive, "next-gen" architectural blueprint for a pcap visualizer. This design elevates the project from a simple script to a scalable, production-grade web application capable of handling the demands of serious network analysis.

The key decisions and recommendations are:

* **Architecture:** A decoupled, service-oriented design 1 modeled after the industry-standard Arkime platform.2 This separates the **Parser**, the **FastAPI** backend, and the **SvelteKit** frontend into independent, high-performance services.  
* **Data:** The implementation leverages **OpenSearch** as a high-speed query engine for all network metadata (SPI).13 This is the "secret" to a fast and responsive UI. For live monitoring, **Server-Sent Events (SSE)** provide a simple, robust, and efficient streaming solution.48  
* **Visuals:** A "next-gen" UI is achieved by assembling a *portfolio* of best-in-class JavaScript libraries, each chosen for its specific strengths:  
  * **Apache ECharts:** For high-performance, large-dataset charts (heatmaps, time-series, geo-maps).80  
  * **Cytoscape.js:** For its powerful, analysis-focused graph theory engine.83  
  * **Vis.js (Timeline):** For its dedicated, interactive timeline component.86

By following this "Golden Path" stack, the result is not just a "web GUI." It is a high-performance, scalable network analysis platform that is performant, visually stunning, and founded on modern architectural principles. The complete deployment guide for Ubuntu 24 provides the actionable, end-to-end path to bring this vision to production.

#### **Works cited**

1. Decoupled architecture \- MiaRec Documentation, accessed November 15, 2025, [https://docs.miarec.com/installation-guide/hardware-requirements/decoupled-architecture/](https://docs.miarec.com/installation-guide/hardware-requirements/decoupled-architecture/)  
2. Arkime is an open source, large scale, full packet capturing, indexing, and database system. \- GitHub, accessed November 15, 2025, [https://github.com/arkime/arkime](https://github.com/arkime/arkime)  
3. Components | Malcolm, accessed November 15, 2025, [https://cisagov.github.io/Malcolm/docs/components.html](https://cisagov.github.io/Malcolm/docs/components.html)  
4. Looking for software that can read .pcap files but that doesn't actually capture data \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/networking/comments/1p5a0o/looking\_for\_software\_that\_can\_read\_pcap\_files\_but/](https://www.reddit.com/r/networking/comments/1p5a0o/looking_for_software_that_can_read_pcap_files_but/)  
5. Arkime FAQ, accessed November 15, 2025, [https://arkime.com/faq](https://arkime.com/faq)  
6. Settings \- Arkime, accessed November 15, 2025, [https://arkime.com/settings](https://arkime.com/settings)  
7. snowflake: PcapXray \- A Network Forensics Tool \- To visualize a Packet Capture offline as a Network Diagram including device identification, highlight important communication and file extraction \- GitHub, accessed November 15, 2025, [https://github.com/Srinivas11789/PcapXray](https://github.com/Srinivas11789/PcapXray)  
8. Extracting the payload from a pcap file using Python | by Vera Worri \- Medium, accessed November 15, 2025, [https://medium.com/@vworri/extracting-the-payload-from-a-pcap-file-using-python-d938d7622d71](https://medium.com/@vworri/extracting-the-payload-from-a-pcap-file-using-python-d938d7622d71)  
9. How to Build a Real-time Network Traffic Dashboard with Python and Streamlit, accessed November 15, 2025, [https://www.freecodecamp.org/news/build-a-real-time-network-traffic-dashboard-with-python-and-streamlit/](https://www.freecodecamp.org/news/build-a-real-time-network-traffic-dashboard-with-python-and-streamlit/)  
10. A Real-Time Streaming System for Customized Network Traffic Capture \- MDPI, accessed November 15, 2025, [https://www.mdpi.com/1424-8220/23/14/6467](https://www.mdpi.com/1424-8220/23/14/6467)  
11. a packet capture and analysis architecture \- Plab \- Unina, accessed November 15, 2025, [http://wpage.unina.it/alberto/papers/TR-DIS-122004.pdf](http://wpage.unina.it/alberto/papers/TR-DIS-122004.pdf)  
12. Packet Analysis — Book of Zeek (8.1.0-dev.682), accessed November 15, 2025, [https://docs.zeek.org/en/master/frameworks/packet-analysis.html](https://docs.zeek.org/en/master/frameworks/packet-analysis.html)  
13. Arkime \- Malcolm.fyi, accessed November 15, 2025, [https://malcolm.fyi/docs/arkime.html](https://malcolm.fyi/docs/arkime.html)  
14. Analyzing network packets with Wireshark, Elasticsearch, and Kibana | Elastic Blog, accessed November 15, 2025, [https://www.elastic.co/blog/analyzing-network-packets-with-wireshark-elasticsearch-and-kibana](https://www.elastic.co/blog/analyzing-network-packets-with-wireshark-elasticsearch-and-kibana)  
15. Real-Time Data Visualization in React using WebSockets and Charts | Syncfusion Blogs, accessed November 15, 2025, [https://www.syncfusion.com/blogs/post/view-real-time-data-using-websocket/amp](https://www.syncfusion.com/blogs/post/view-real-time-data-using-websocket/amp)  
16. Building a Real-time Dashboard with FastAPI and Svelte | TestDriven.io, accessed November 15, 2025, [https://testdriven.io/blog/fastapi-svelte/](https://testdriven.io/blog/fastapi-svelte/)  
17. Stream Data to Charts in Real-Time using Websockets and Express \- DEV Community, accessed November 15, 2025, [https://dev.to/manoj\_004d/stream-data-to-charts-in-real-time-using-websockets-and-express-173b](https://dev.to/manoj_004d/stream-data-to-charts-in-real-time-using-websockets-and-express-173b)  
18. Real-time data visualization with WebSocket | by Niilo Keinänen | Medium, accessed November 15, 2025, [https://niilo-keinanen-93801.medium.com/real-time-data-visualization-with-websocket-79773edbf477](https://niilo-keinanen-93801.medium.com/real-time-data-visualization-with-websocket-79773edbf477)  
19. Mastering PCAP Review: Essential Guide for Analysts & Engineers \- Insane Cyber, accessed November 15, 2025, [https://insanecyber.com/mastering-pcap-review/](https://insanecyber.com/mastering-pcap-review/)  
20. Arkime, accessed November 15, 2025, [https://arkime.com/](https://arkime.com/)  
21. Querying Like a Pro in Arkime: Getting the Most Out of Arkime Viewer: Beyond the Basics, accessed November 15, 2025, [https://medium.com/@cyberengage.org/querying-like-a-pro-in-arkime-getting-the-most-out-of-arkime-viewer-beyond-the-basics-4452dce482e4](https://medium.com/@cyberengage.org/querying-like-a-pro-in-arkime-getting-the-most-out-of-arkime-viewer-beyond-the-basics-4452dce482e4)  
22. Introduction to Node.js, accessed November 15, 2025, [https://nodejs.org/en/learn/getting-started/introduction-to-nodejs](https://nodejs.org/en/learn/getting-started/introduction-to-nodejs)  
23. FastAPI vs Node.js for Building APIs \- planeks, accessed November 15, 2025, [https://www.planeks.net/nodejs-vs-fastapi-for-api/](https://www.planeks.net/nodejs-vs-fastapi-for-api/)  
24. FastAPI or Node? : r/webdev \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/webdev/comments/1kqbymf/fastapi\_or\_node/](https://www.reddit.com/r/webdev/comments/1kqbymf/fastapi_or_node/)  
25. 7 Best Web Development Stacks to Use in 2025 \- WPWeb Elite, accessed November 15, 2025, [https://www.wpwebelite.com/blog/web-development-stacks/](https://www.wpwebelite.com/blog/web-development-stacks/)  
26. Pros and Cons of Flask and FastAPI (When to Use Each?) | by Shariq Ahmed \- Medium, accessed November 15, 2025, [https://medium.com/@shariq.ahmed525/pros-and-cons-of-flask-and-fastapi-when-to-use-each-b987ee89cf96](https://medium.com/@shariq.ahmed525/pros-and-cons-of-flask-and-fastapi-when-to-use-each-b987ee89cf96)  
27. FastAPI vs Flask: Key Differences, Performance, and Use Cases \- Codecademy, accessed November 15, 2025, [https://www.codecademy.com/article/fastapi-vs-flask-key-differences-performance-and-use-cases](https://www.codecademy.com/article/fastapi-vs-flask-key-differences-performance-and-use-cases)  
28. Why FastAPI Became My Preferred Backend Framework Over Node.js: My In-Depth Journey, accessed November 15, 2025, [https://rajeevbarnwal.medium.com/why-fastapi-became-my-preferred-backend-framework-over-node-js-my-in-depth-journey-e198c0c59fda?source=rss-514f23f56c32------2](https://rajeevbarnwal.medium.com/why-fastapi-became-my-preferred-backend-framework-over-node-js-my-in-depth-journey-e198c0c59fda?source=rss-514f23f56c32------2)  
29. Battle of the Backends: FastAPI vs. Node.js \- HostAdvice, accessed November 15, 2025, [https://hostadvice.com/blog/web-hosting/node-js/fastapi-vs-nodejs/](https://hostadvice.com/blog/web-hosting/node-js/fastapi-vs-nodejs/)  
30. Flask vs fastapi : r/flask \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/flask/comments/13pyxie/flask\_vs\_fastapi/](https://www.reddit.com/r/flask/comments/13pyxie/flask_vs_fastapi/)  
31. Top 7 Frontend Frameworks to Use in 2025: Pro Advice \- Developer Roadmaps, accessed November 15, 2025, [https://roadmap.sh/frontend/frameworks](https://roadmap.sh/frontend/frameworks)  
32. Best 19 React UI Component Libraries in 2025 \- Prismic, accessed November 15, 2025, [https://prismic.io/blog/react-component-libraries](https://prismic.io/blog/react-component-libraries)  
33. Frontend Faceoff 2025: React vs Vue vs Svelte \- Skylinkindia, accessed November 15, 2025, [https://skylinkindia.in/frontend-faceoff-2025-react-vs-vue-vs-svelte/](https://skylinkindia.in/frontend-faceoff-2025-react-vs-vue-vs-svelte/)  
34. When to choose React over Svelte : r/sveltejs \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/sveltejs/comments/1jeknib/when\_to\_choose\_react\_over\_svelte/](https://www.reddit.com/r/sveltejs/comments/1jeknib/when_to_choose_react_over_svelte/)  
35. React vs Vue vs Svelte: Choosing the Right Framework for 2025 \- Medium, accessed November 15, 2025, [https://medium.com/@ignatovich.dm/react-vs-vue-vs-svelte-choosing-the-right-framework-for-2025-4f4bb9da35b4](https://medium.com/@ignatovich.dm/react-vs-vue-vs-svelte-choosing-the-right-framework-for-2025-4f4bb9da35b4)  
36. Why Learn Svelte in 2025? The Value Proposition & Svelte vs React & Vue, accessed November 15, 2025, [https://dev.to/a1guy/why-learn-svelte-in-2025-the-value-proposition-svelte-vs-react-vue-1bhc](https://dev.to/a1guy/why-learn-svelte-in-2025-the-value-proposition-svelte-vs-react-vue-1bhc)  
37. Svelte vs React: Technical & Business Comparison \[2026\] \- The Frontend Company, accessed November 15, 2025, [https://www.thefrontendcompany.com/posts/svelte-vs-react](https://www.thefrontendcompany.com/posts/svelte-vs-react)  
38. Svelte vs. React 2025 Comparison, Which is Better \- Aalpha Information Systems, accessed November 15, 2025, [https://www.aalpha.net/articles/svelte-vs-react-comparison/](https://www.aalpha.net/articles/svelte-vs-react-comparison/)  
39. SvelteKit vs Next.js | Better Stack Community, accessed November 15, 2025, [https://betterstack.com/community/guides/scaling-nodejs/sveltekit-vs-nextjs/](https://betterstack.com/community/guides/scaling-nodejs/sveltekit-vs-nextjs/)  
40. jasongitmail/svelte-vs-next: Comparison of major features in SvelteKit vs NextJS. \- GitHub, accessed November 15, 2025, [https://github.com/jasongitmail/svelte-vs-next](https://github.com/jasongitmail/svelte-vs-next)  
41. Svelte vs. Next.js: Understand the Differences \- Descope, accessed November 15, 2025, [https://www.descope.com/blog/post/nextjs-vs-reactjs-vs-sveltekit](https://www.descope.com/blog/post/nextjs-vs-reactjs-vs-sveltekit)  
42. Sveltekit vs. Next.js: A side-by-side comparison | Hygraph, accessed November 15, 2025, [https://hygraph.com/blog/sveltekit-vs-nextjs](https://hygraph.com/blog/sveltekit-vs-nextjs)  
43. Server-Sent Events vs WebSockets – How to Choose a Real-Time Data Exchange Protocol, accessed November 15, 2025, [https://www.freecodecamp.org/news/server-sent-events-vs-websockets/](https://www.freecodecamp.org/news/server-sent-events-vs-websockets/)  
44. SSE vs WebSockets: Comparing Real-Time Communication Protocols \- SoftwareMill, accessed November 15, 2025, [https://softwaremill.com/sse-vs-websockets-comparing-real-time-communication-protocols/](https://softwaremill.com/sse-vs-websockets-comparing-real-time-communication-protocols/)  
45. Real-Time Features: WebSockets vs. Server-Sent Events vs. Polling \- Medium, accessed November 15, 2025, [https://medium.com/towardsdev/real-time-features-websockets-vs-server-sent-events-vs-polling-e7b3d07e6442](https://medium.com/towardsdev/real-time-features-websockets-vs-server-sent-events-vs-polling-e7b3d07e6442)  
46. Data Visualization Real Time with WebSockets & LightningChart JS, accessed November 15, 2025, [https://lightningchart.com/blog/data-visualization-websockets/](https://lightningchart.com/blog/data-visualization-websockets/)  
47. WebSocket vs REST: Key differences and which to use \- Ably, accessed November 15, 2025, [https://ably.com/topic/websocket-vs-rest](https://ably.com/topic/websocket-vs-rest)  
48. WebSockets vs Server-Sent Events (SSE): Choosing Your Real-Time Protocol, accessed November 15, 2025, [https://websocket.org/comparisons/sse/](https://websocket.org/comparisons/sse/)  
49. WebSockets vs Server-Sent Events: Key differences and which to use in 2024 \- Ably, accessed November 15, 2025, [https://ably.com/blog/websockets-vs-sse](https://ably.com/blog/websockets-vs-sse)  
50. WebSocket vs REST API: How WebSocket Improves Real-Time Monitoring Performance by 98.5% | by Arif Rahman | Python in Plain English, accessed November 15, 2025, [https://python.plainenglish.io/websocket-vs-rest-api-how-websocket-improves-real-time-monitoring-performance-by-98-5-822fcf4af6bf](https://python.plainenglish.io/websocket-vs-rest-api-how-websocket-improves-real-time-monitoring-performance-by-98-5-822fcf4af6bf)  
51. Real-Time Notifications in Python: Using SSE with FastAPI | by İnan DELİBAŞ | Medium, accessed November 15, 2025, [https://medium.com/@inandelibas/real-time-notifications-in-python-using-sse-with-fastapi-1c8c54746eb7](https://medium.com/@inandelibas/real-time-notifications-in-python-using-sse-with-fastapi-1c8c54746eb7)  
52. WebSockets vs. Server-Sent events/EventSource \[closed\] \- Stack Overflow, accessed November 15, 2025, [https://stackoverflow.com/questions/5195452/websockets-vs-server-sent-events-eventsource](https://stackoverflow.com/questions/5195452/websockets-vs-server-sent-events-eventsource)  
53. What is the fastest way to send messages: websockets or server sent events (SSE)?, accessed November 15, 2025, [https://www.reddit.com/r/webdev/comments/1nngnlb/what\_is\_the\_fastest\_way\_to\_send\_messages/](https://www.reddit.com/r/webdev/comments/1nngnlb/what_is_the_fastest_way_to_send_messages/)  
54. Streaming: Websockets vs SSE? : r/Rag \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/Rag/comments/1f721qu/streaming\_websockets\_vs\_sse/](https://www.reddit.com/r/Rag/comments/1f721qu/streaming_websockets_vs_sse/)  
55. Streaming HTTP vs. WebSocket vs. SSE: A Comparison for Real-Time Data, accessed November 15, 2025, [https://dev.to/mechcloud\_academy/streaming-http-vs-websocket-vs-sse-a-comparison-for-real-time-data-1geo](https://dev.to/mechcloud_academy/streaming-http-vs-websocket-vs-sse-a-comparison-for-real-time-data-1geo)  
56. Performance difference between websocket and server sent events (SSE) for chat room for ... \- Stack Overflow, accessed November 15, 2025, [https://stackoverflow.com/questions/63583989/performance-difference-between-websocket-and-server-sent-events-sse-for-chat-r](https://stackoverflow.com/questions/63583989/performance-difference-between-websocket-and-server-sent-events-sse-for-chat-r)  
57. WebSockets \- FastAPI, accessed November 15, 2025, [https://fastapi.tiangolo.com/advanced/websockets/](https://fastapi.tiangolo.com/advanced/websockets/)  
58. Browse thousands of Network Monitoring images for design inspiration \- Dribbble, accessed November 15, 2025, [https://dribbble.com/search/network-monitoring](https://dribbble.com/search/network-monitoring)  
59. Cyber Security designs, themes, templates and downloadable graphic elements on Dribbble, accessed November 15, 2025, [https://dribbble.com/tags/cyber-security?page=2](https://dribbble.com/tags/cyber-security?page=2)  
60. Dark Dashboard Graphic Templates \- Envato, accessed November 15, 2025, [https://elements.envato.com/graphic-templates/dark+dashboard](https://elements.envato.com/graphic-templates/dark+dashboard)  
61. Dark Mode Dashboard \- Dribbble, accessed November 15, 2025, [https://dribbble.com/tags/dark-mode-dashboard](https://dribbble.com/tags/dark-mode-dashboard)  
62. Dark Dashboard designs, themes, templates and downloadable graphic elements on Dribbble, accessed November 15, 2025, [https://dribbble.com/tags/dark-dashboard](https://dribbble.com/tags/dark-dashboard)  
63. Dark Dashboard Ui royalty-free images \- Shutterstock, accessed November 15, 2025, [https://www.shutterstock.com/search/dark-dashboard-ui](https://www.shutterstock.com/search/dark-dashboard-ui)  
64. 60+ Dark mode screen design examples | Muzli Design Inspiration, accessed November 15, 2025, [https://muz.li/inspiration/dark-mode/](https://muz.li/inspiration/dark-mode/)  
65. Cybersecurity Dashboard designs, themes, templates and downloadable graphic elements on Dribbble, accessed November 15, 2025, [https://dribbble.com/tags/cybersecurity-dashboard](https://dribbble.com/tags/cybersecurity-dashboard)  
66. Browse thousands of Cybersecurity Dashboard images for design inspiration \- Dribbble, accessed November 15, 2025, [https://dribbble.com/search/cybersecurity-dashboard](https://dribbble.com/search/cybersecurity-dashboard)  
67. Dashboard Design: best practices and examples \- Justinmind, accessed November 15, 2025, [https://www.justinmind.com/ui-design/dashboard-design-best-practices-ux](https://www.justinmind.com/ui-design/dashboard-design-best-practices-ux)  
68. Cisco Meraki Next Gen Dashboard | Smarter UI for Scalable Network Management, accessed November 15, 2025, [https://www.youtube.com/watch?v=\_fmsLNQSZrQ](https://www.youtube.com/watch?v=_fmsLNQSZrQ)  
69. Introducing a new look to the Meraki dashboard and mobile app, accessed November 15, 2025, [https://community.meraki.com/t5/Feature-Announcements/Introducing-a-new-look-to-the-Meraki-dashboard-and-mobile-app/ba-p/273803](https://community.meraki.com/t5/Feature-Announcements/Introducing-a-new-look-to-the-Meraki-dashboard-and-mobile-app/ba-p/273803)  
70. Work | Cisco Meraki Next Generation Dashboard Visiontype \- DesignMap, accessed November 15, 2025, [https://designmap.com/work/ciscomeraki](https://designmap.com/work/ciscomeraki)  
71. MUI: The React component library you always wanted, accessed November 15, 2025, [https://mui.com/](https://mui.com/)  
72. The Ultimate Comparison Between Ant Design Vs Material UI | Magic UI, accessed November 15, 2025, [https://magicui.design/blog/ant-design-vs-material-ui](https://magicui.design/blog/ant-design-vs-material-ui)  
73. 14 Best React UI Component Libraries in 2025 (+ Alternatives to MUI & Shadcn) | Untitled UI, accessed November 15, 2025, [https://www.untitledui.com/blog/react-component-libraries](https://www.untitledui.com/blog/react-component-libraries)  
74. Ant Design \+ Tailwind CSS or alternative : r/reactjs \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/reactjs/comments/1ju189l/ant\_design\_tailwind\_css\_or\_alternative/](https://www.reddit.com/r/reactjs/comments/1ju189l/ant_design_tailwind_css_or_alternative/)  
75. Top 15 Data Visualization Frameworks \- GeeksforGeeks, accessed November 15, 2025, [https://www.geeksforgeeks.org/data-visualization/top-15-data-visualization-frameworks/](https://www.geeksforgeeks.org/data-visualization/top-15-data-visualization-frameworks/)  
76. Most "future-proof" data visualisation frontend framework to learn in 2024? \- Reddit, accessed November 15, 2025, [https://www.reddit.com/r/webdev/comments/1cxy2dt/most\_futureproof\_data\_visualisation\_frontend/](https://www.reddit.com/r/webdev/comments/1cxy2dt/most_futureproof_data_visualisation_frontend/)  
77. D3 by Observable | The JavaScript library for bespoke data visualization, accessed November 15, 2025, [https://d3js.org/](https://d3js.org/)  
78. 7 best JavaScript Chart Libraries (and a Faster Alternative for 2026\) \- Luzmo, accessed November 15, 2025, [https://www.luzmo.com/blog/best-javascript-chart-libraries](https://www.luzmo.com/blog/best-javascript-chart-libraries)  
79. JavaScript Chart Libraries In 2026: Best Picks \+ Alternatives | Luzmo, accessed November 15, 2025, [https://www.luzmo.com/blog/javascript-chart-libraries](https://www.luzmo.com/blog/javascript-chart-libraries)  
80. Apache ECharts, accessed November 15, 2025, [https://echarts.apache.org/](https://echarts.apache.org/)  
81. 6 Best JavaScript Charting Libraries for Dashboards in 2026 \- Embeddable, accessed November 15, 2025, [https://embeddable.com/blog/javascript-charting-libraries](https://embeddable.com/blog/javascript-charting-libraries)  
82. Top 10 JavaScript Charting Libraries in 2025 \- Carmatec, accessed November 15, 2025, [https://www.carmatec.com/blog/top-10-javascript-charting-libraries/](https://www.carmatec.com/blog/top-10-javascript-charting-libraries/)  
83. Cytoscape.js, accessed November 15, 2025, [https://js.cytoscape.org/](https://js.cytoscape.org/)  
84. What is Cytoscape?, accessed November 15, 2025, [https://cytoscape.org/what\_is\_cytoscape.html](https://cytoscape.org/what_is_cytoscape.html)  
85. reveal.js \- Cytoscape, accessed November 15, 2025, [https://cytoscape.org/cytoscape-tutorials/presentations/modules/network-analysis/index.html](https://cytoscape.org/cytoscape-tutorials/presentations/modules/network-analysis/index.html)  
86. vis.js, accessed November 15, 2025, [https://visjs.org/](https://visjs.org/)  
87. timeline \- vis.js \- A dynamic, browser based visualization library. \- GitHub Pages, accessed November 15, 2025, [https://visjs.github.io/vis-timeline/docs/timeline/](https://visjs.github.io/vis-timeline/docs/timeline/)  
88. Network Topology \- Cisco Meraki Documentation, accessed November 15, 2025, [https://documentation.meraki.com/Switching/MS\_-\_Switches/Operate\_and\_Maintain/Monitoring\_and\_Reporting/Network\_Topology](https://documentation.meraki.com/Switching/MS_-_Switches/Operate_and_Maintain/Monitoring_and_Reporting/Network_Topology)  
89. Sigma.js, accessed November 15, 2025, [https://www.sigmajs.org/](https://www.sigmajs.org/)  
90. Top 13 JavaScript graph visualization libraries \- Linkurious, accessed November 15, 2025, [https://linkurious.com/blog/top-javascript-graph-libraries/](https://linkurious.com/blog/top-javascript-graph-libraries/)  
91. vis.js \- Network documentation. \- GitHub Pages, accessed November 15, 2025, [https://visjs.github.io/vis-network/docs/](https://visjs.github.io/vis-network/docs/)  
92. d3-force | D3 by Observable \- D3.js, accessed November 15, 2025, [https://d3js.org/d3-force](https://d3js.org/d3-force)  
93. 11 Bonus Chapter: Advanced Network Visualization using D3, accessed November 15, 2025, [https://ona-book.org/advanced-viz.html](https://ona-book.org/advanced-viz.html)  
94. 8.19. Flow Graph \- Wireshark, accessed November 15, 2025, [https://www.wireshark.org/docs/wsug\_html\_chunked/ChStatFlowGraph.html](https://www.wireshark.org/docs/wsug_html_chunked/ChStatFlowGraph.html)  
95. Flow Graph in Wireshark \- GeeksforGeeks, accessed November 15, 2025, [https://www.geeksforgeeks.org/ethical-hacking/flow-graph-in-wireshark/](https://www.geeksforgeeks.org/ethical-hacking/flow-graph-in-wireshark/)  
96. visjs/vis-timeline: Create a fully customizable, interactive timelines and 2d-graphs with items and ranges. \- GitHub, accessed November 15, 2025, [https://github.com/visjs/vis-timeline](https://github.com/visjs/vis-timeline)  
97. Display a Workflow As a Timeline Using Vis.js | Aras, accessed November 15, 2025, [https://aras.com/en/blog/display-workflow-timeline-using-vis-js](https://aras.com/en/blog/display-workflow-timeline-using-vis-js)  
98. How to Keep a Record of Network Performance Over Time \- PingPlotter, accessed November 15, 2025, [https://www.pingplotter.com/wisdom/article/historical-data/](https://www.pingplotter.com/wisdom/article/historical-data/)  
99. Metric graphs 101: Timeseries graphs \- Datadog, accessed November 15, 2025, [https://www.datadoghq.com/blog/timeseries-metric-graphs-101/](https://www.datadoghq.com/blog/timeseries-metric-graphs-101/)  
100. Essential Chart Types for Data Visualization | Atlassian, accessed November 15, 2025, [https://www.atlassian.com/data/charts/essential-chart-types-for-data-visualization](https://www.atlassian.com/data/charts/essential-chart-types-for-data-visualization)  
101. Data visualization \- Material Design 2, accessed November 15, 2025, [https://m2.material.io/design/communication/data-visualization.html](https://m2.material.io/design/communication/data-visualization.html)  
102. Features \- Apache ECharts, accessed November 15, 2025, [https://echarts.apache.org/en/feature.html](https://echarts.apache.org/en/feature.html)  
103. D3.js vs ECharts: When to Go Custom, When to Go Fast | by Sanjay Nelagadde \- Medium, accessed November 15, 2025, [https://medium.com/data-science-collective/d3-js-vs-echarts-when-to-go-custom-when-to-go-fast-2a922c9956b6](https://medium.com/data-science-collective/d3-js-vs-echarts-when-to-go-custom-when-to-go-fast-2a922c9956b6)  
104. Examples \- Apache ECharts, accessed November 15, 2025, [https://echarts.apache.org/examples/en/index.html](https://echarts.apache.org/examples/en/index.html)  
105. data visualization \- Echarts 3 \- Bipartite chart \- Stack Overflow, accessed November 15, 2025, [https://stackoverflow.com/questions/46667154/echarts-3-bipartite-chart](https://stackoverflow.com/questions/46667154/echarts-3-bipartite-chart)  
106. Visualizing Time Series Data with ECharts and InfluxDB, accessed November 15, 2025, [https://www.influxdata.com/blog/visualizing-time-series-data-echarts-influxdb/](https://www.influxdata.com/blog/visualizing-time-series-data-echarts-influxdb/)  
107. 80 types of charts & graphs for data visualization (with examples) \- Datylon, accessed November 15, 2025, [https://www.datylon.com/blog/types-of-charts-graphs-examples-data-visualization](https://www.datylon.com/blog/types-of-charts-graphs-examples-data-visualization)  
108. Chart types \- UNHCR Dataviz Platform, accessed November 15, 2025, [https://dataviz.unhcr.org/chart-types/](https://dataviz.unhcr.org/chart-types/)  
109. Plotting networks — Network Data Science \- Benjamin D. Pedigo, accessed November 15, 2025, [https://bdpedigo.github.io/networks-course/plotting\_networks.html](https://bdpedigo.github.io/networks-course/plotting_networks.html)  
110. Interactive Web-Based Visual Analysis on Network Traffic Data \- MDPI, accessed November 15, 2025, [https://www.mdpi.com/2078-2489/14/1/16](https://www.mdpi.com/2078-2489/14/1/16)  
111. Heat map \- Wikipedia, accessed November 15, 2025, [https://en.wikipedia.org/wiki/Heat\_map](https://en.wikipedia.org/wiki/Heat_map)  
112. Visualizing Correlations: Scatter Matrix and Heat map | by Becaye Baldé \- Medium, accessed November 15, 2025, [https://medium.com/@becaye-balde/visualizing-correlations-scatter-matrix-and-heat-map-d597436b7d23](https://medium.com/@becaye-balde/visualizing-correlations-scatter-matrix-and-heat-map-d597436b7d23)  
113. Heatmap of Correlation Matrix | CodeSignal Learn, accessed November 15, 2025, [https://codesignal.com/learn/courses/feature-engineering-and-correlation-analysis-in-pandas/lessons/heatmap-of-correlation-matrix](https://codesignal.com/learn/courses/feature-engineering-and-correlation-analysis-in-pandas/lessons/heatmap-of-correlation-matrix)  
114. Visualize network traffic patterns by using open-source tools \- Azure Network Watcher, accessed November 15, 2025, [https://learn.microsoft.com/en-us/azure/network-watcher/network-watcher-using-open-source-tools](https://learn.microsoft.com/en-us/azure/network-watcher/network-watcher-using-open-source-tools)  
115. CapAnalysis | PCAP from another point of view, accessed November 15, 2025, [https://www.capanalysis.net/](https://www.capanalysis.net/)  
116. CapAnalysis \- Deep Packet Inspection | PPTX \- Slideshare, accessed November 15, 2025, [https://www.slideshare.net/slideshow/capanalysis-capture-analysis-deep-packet-inspection/38863642](https://www.slideshare.net/slideshow/capanalysis-capture-analysis-deep-packet-inspection/38863642)  
117. Deploying a FastAPI project on Ubuntu 24 | Simple Information Inc., accessed November 15, 2025, [https://www.simpleinformation.com/article/deploying-fastapi-project-ubuntu-24](https://www.simpleinformation.com/article/deploying-fastapi-project-ubuntu-24)  
118. Deploying a FastAPI Project to an Ubuntu VPS — A Complete Guide for Developers \- DEV Community, accessed November 15, 2025, [https://dev.to/1amkaizen/deploying-a-fastapi-project-to-an-ubuntu-vps-a-complete-guide-for-developers-392](https://dev.to/1amkaizen/deploying-a-fastapi-project-to-an-ubuntu-vps-a-complete-guide-for-developers-392)  
119. How to Install Node.js on Ubuntu (Step-by-Step Guide) | DigitalOcean, accessed November 15, 2025, [https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-22-04](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-22-04)  
120. Beginner's Guide for Containerizing Application — Deploying a Full-Stack FastAPI and React App with Docker and NGINX on Local \- Abhishek Jain, accessed November 15, 2025, [https://vardhmanandroid2015.medium.com/beginners-guide-for-containerizing-application-deploying-a-full-stack-fastapi-and-react-app-001f2cac08a8](https://vardhmanandroid2015.medium.com/beginners-guide-for-containerizing-application-deploying-a-full-stack-fastapi-and-react-app-001f2cac08a8)