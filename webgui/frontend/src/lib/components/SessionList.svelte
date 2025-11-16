<script>
	import { onMount } from 'svelte';
	import { getFlows } from '$lib/api/client.js';

	export let jobId;

	let flows = [];
	let total = 0;
	let loading = true;
	let error = null;

	let filters = {
		src_ip: '',
		dst_ip: '',
		protocol: ''
	};

	let currentPage = 1;
	let pageSize = 50;

	onMount(async () => {
		await loadFlows();
	});

	async function loadFlows() {
		loading = true;
		try {
			const params = {
				limit: pageSize,
				offset: (currentPage - 1) * pageSize
			};

			// Add filters if set
			if (filters.src_ip) params.src_ip = filters.src_ip;
			if (filters.dst_ip) params.dst_ip = filters.dst_ip;
			if (filters.protocol) params.protocol = filters.protocol;

			const data = await getFlows(jobId, params);
			flows = data.flows;
			total = data.total;
		} catch (err) {
			error = err.message;
		} finally {
			loading = false;
		}
	}

	function formatBytes(bytes) {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
	}

	function formatDuration(ms) {
		if (ms < 1000) return `${ms}ms`;
		if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
		return `${(ms / 60000).toFixed(1)}m`;
	}

	async function applyFilters() {
		currentPage = 1;
		await loadFlows();
	}

	function clearFilters() {
		filters = { src_ip: '', dst_ip: '', protocol: '' };
		applyFilters();
	}

	async function nextPage() {
		if (currentPage * pageSize < total) {
			currentPage++;
			await loadFlows();
		}
	}

	async function previousPage() {
		if (currentPage > 1) {
			currentPage--;
			await loadFlows();
		}
	}
</script>

<div class="card bg-base-200 shadow-xl">
	<div class="card-body">
		<h2 class="card-title mb-4">Network Sessions</h2>

		<!-- Filters -->
		<div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
			<input
				type="text"
				placeholder="Source IP"
				class="input input-bordered"
				bind:value={filters.src_ip}
			/>
			<input
				type="text"
				placeholder="Destination IP"
				class="input input-bordered"
				bind:value={filters.dst_ip}
			/>
			<select class="select select-bordered" bind:value={filters.protocol}>
				<option value="">All Protocols</option>
				<option value="TCP">TCP</option>
				<option value="UDP">UDP</option>
				<option value="ICMP">ICMP</option>
			</select>
			<div class="flex gap-2">
				<button class="btn btn-primary flex-1" on:click={applyFilters}>Apply</button>
				<button class="btn btn-ghost" on:click={clearFilters}>Clear</button>
			</div>
		</div>

		{#if loading}
			<div class="flex justify-center items-center h-[400px]">
				<span class="loading loading-spinner loading-lg text-primary"></span>
			</div>
		{:else if error}
			<div class="alert alert-error">
				<span>Error loading sessions: {error}</span>
			</div>
		{:else if flows.length === 0}
			<div class="text-center text-base-content/70 py-8">
				No sessions found matching your filters.
			</div>
		{:else}
			<div class="overflow-x-auto">
				<table class="table table-zebra table-sm">
					<thead>
						<tr>
							<th>Source IP</th>
							<th>Port</th>
							<th>Destination IP</th>
							<th>Port</th>
							<th>Protocol</th>
							<th>Packets</th>
							<th>Bytes</th>
							<th>Duration</th>
						</tr>
					</thead>
					<tbody>
						{#each flows as flow}
							<tr class="hover">
								<td class="font-mono text-xs">{flow.src_ip}</td>
								<td>{flow.src_port}</td>
								<td class="font-mono text-xs">{flow.dst_ip}</td>
								<td>{flow.dst_port}</td>
								<td>
									<span class="badge badge-sm">{flow.protocol}</span>
								</td>
								<td>{flow.packet_count.toLocaleString()}</td>
								<td>{formatBytes(flow.total_bytes)}</td>
								<td>{formatDuration(flow.duration_ms)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>

			<!-- Pagination -->
			<div class="flex justify-between items-center mt-4">
				<div class="text-sm text-base-content/70">
					Showing {(currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, total)} of {total}
					sessions
				</div>
				<div class="btn-group">
					<button class="btn btn-sm" on:click={previousPage} disabled={currentPage === 1}>
						«
					</button>
					<button class="btn btn-sm btn-disabled">Page {currentPage}</button>
					<button class="btn btn-sm" on:click={nextPage} disabled={currentPage * pageSize >= total}>
						»
					</button>
				</div>
			</div>
		{/if}
	</div>
</div>
