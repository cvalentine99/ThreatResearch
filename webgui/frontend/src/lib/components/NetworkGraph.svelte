<script>
	import { onMount } from 'svelte';
	import { getNetworkTopology } from '$lib/api/client.js';
	import cytoscape from 'cytoscape';

	export let jobId;

	let container;
	let cy;
	let loading = true;
	let error = null;
	let stats = { nodes: 0, edges: 0 };

	onMount(async () => {
		await loadGraph();
	});

	async function loadGraph() {
		try {
			console.log('NetworkGraph: Loading topology for job', jobId);
			const data = await getNetworkTopology(jobId, 500);
			console.log('NetworkGraph: Got topology data', data);

			stats = {
				nodes: data.total_nodes,
				edges: data.total_edges
			};

			// Initialize Cytoscape
			cy = cytoscape({
				container: container,
				elements: [...data.nodes, ...data.edges],
				style: [
					{
						selector: 'node',
						style: {
							'background-color': (node) => {
								return node.data('type') === 'internal' ? '#3b82f6' : '#8b5cf6';
							},
							label: 'data(label)',
							'text-valign': 'center',
							'text-halign': 'center',
							color: '#fff',
							'font-size': '10px',
							width: 30,
							height: 30
						}
					},
					{
						selector: 'edge',
						style: {
							width: (edge) => {
								const packets = edge.data('packet_count');
								return Math.min(5, Math.max(1, Math.log(packets)));
							},
							'line-color': '#22d3ee',
							'target-arrow-color': '#22d3ee',
							'target-arrow-shape': 'triangle',
							'curve-style': 'bezier',
							opacity: 0.6
						}
					}
				],
				layout: {
					name: 'cose',
					idealEdgeLength: 100,
					nodeOverlap: 20,
					refresh: 20,
					fit: true,
					padding: 30,
					randomize: false,
					componentSpacing: 100,
					nodeRepulsion: 400000,
					edgeElasticity: 100,
					nestingFactor: 5,
					gravity: 80,
					numIter: 1000,
					initialTemp: 200,
					coolingFactor: 0.95,
					minTemp: 1.0
				},
				minZoom: 0.1,
				maxZoom: 10,
				wheelSensitivity: 0.2
			});

			// Add event listeners
			cy.on('tap', 'node', (event) => {
				const node = event.target;
				console.log('Clicked node:', node.data('label'));
			});

			cy.on('tap', 'edge', (event) => {
				const edge = event.target;
				const data = edge.data();
				console.log('Clicked edge:', data.protocol, data.packet_count, 'packets');
			});

			loading = false;
		} catch (err) {
			error = err.message;
			loading = false;
		}
	}

	function resetView() {
		if (cy) {
			cy.fit(null, 50);
		}
	}

	function zoomIn() {
		if (cy) {
			cy.zoom(cy.zoom() * 1.2);
			cy.center();
		}
	}

	function zoomOut() {
		if (cy) {
			cy.zoom(cy.zoom() * 0.8);
			cy.center();
		}
	}
</script>

<div class="card bg-base-200 shadow-xl">
	<div class="card-body">
		<div class="flex items-center justify-between mb-4">
			<h2 class="card-title">Network Topology</h2>

			<div class="flex gap-2">
				<button class="btn btn-sm btn-ghost" on:click={zoomIn} title="Zoom In">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-5 w-5"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
					</svg>
				</button>
				<button class="btn btn-sm btn-ghost" on:click={zoomOut} title="Zoom Out">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-5 w-5"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4" />
					</svg>
				</button>
				<button class="btn btn-sm btn-primary" on:click={resetView}>
					Reset View
				</button>
			</div>
		</div>

		{#if loading}
			<div class="flex justify-center items-center h-[600px]">
				<span class="loading loading-spinner loading-lg text-primary"></span>
			</div>
		{:else if error}
			<div class="alert alert-error">
				<span>Error loading graph: {error}</span>
			</div>
		{:else}
			<div class="grid grid-cols-2 gap-4 mb-4">
				<div class="stat bg-base-300 rounded-lg">
					<div class="stat-title">Nodes (IPs)</div>
					<div class="stat-value text-primary">{stats.nodes}</div>
				</div>
				<div class="stat bg-base-300 rounded-lg">
					<div class="stat-title">Edges (Flows)</div>
					<div class="stat-value text-secondary">{stats.edges}</div>
				</div>
			</div>

			<div class="flex gap-4 mb-2">
				<div class="flex items-center gap-2">
					<div class="w-4 h-4 rounded-full bg-primary"></div>
					<span class="text-sm">Internal IPs</span>
				</div>
				<div class="flex items-center gap-2">
					<div class="w-4 h-4 rounded-full bg-secondary"></div>
					<span class="text-sm">External IPs</span>
				</div>
			</div>

			<div bind:this={container} class="w-full h-[600px] bg-base-300 rounded-lg"></div>
		{/if}
	</div>
</div>

<style>
	:global(.cytoscape-container) {
		width: 100%;
		height: 100%;
	}
</style>
