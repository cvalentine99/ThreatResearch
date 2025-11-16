<script>
	import { onMount, onDestroy } from 'svelte';
	import { page } from '$app/stores';
	import { getJobStatus, getJobSummary, streamJobStatus } from '$lib/api/client.js';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import StatsCharts from '$lib/components/StatsCharts.svelte';
	import SessionList from '$lib/components/SessionList.svelte';

	$: jobId = $page.params.jobId;

	let job = null;
	let summary = null;
	let loading = true;
	let error = null;
	let eventSource = null;
	let activeTab = 'overview';

	onMount(async () => {
		await loadJobStatus();

		// If job is running, start streaming updates
		if (job && job.status === 'running') {
			startStreaming();
		} else if (job && job.status === 'completed') {
			await loadSummary();
		}
	});

	onDestroy(() => {
		if (eventSource) {
			eventSource.close();
		}
	});

	async function loadJobStatus() {
		try {
			job = await getJobStatus(jobId);
			loading = false;
		} catch (err) {
			error = err.message;
			loading = false;
		}
	}

	async function loadSummary() {
		try {
			summary = await getJobSummary(jobId);
		} catch (err) {
			console.error('Failed to load summary:', err);
		}
	}

	function startStreaming() {
		eventSource = streamJobStatus(jobId);

		eventSource.onmessage = async (event) => {
			const data = JSON.parse(event.data);

			if (data.error) {
				console.error('Stream error:', data.error);
				eventSource.close();
				return;
			}

			// Update job status
			job = { ...job, ...data };

			// If completed, load summary and close stream
			if (data.status === 'completed') {
				await loadSummary();
				eventSource.close();
				eventSource = null;
			} else if (data.status === 'failed') {
				error = 'Parsing failed';
				eventSource.close();
				eventSource = null;
			}
		};

		eventSource.onerror = (err) => {
			console.error('EventSource error:', err);
			eventSource.close();
			eventSource = null;
		};
	}

	function getStatusColor(status) {
		const colors = {
			pending: 'text-warning',
			running: 'text-info',
			completed: 'text-success',
			failed: 'text-error'
		};
		return colors[status] || 'text-base-content';
	}
</script>

<svelte:head>
	<title>Dashboard - {job?.filename || 'Loading...'}</title>
</svelte:head>

<div class="animate-fade-in">
	{#if loading}
		<div class="flex justify-center items-center min-h-[400px]">
			<span class="loading loading-spinner loading-lg text-primary"></span>
		</div>
	{:else if error}
		<div class="alert alert-error">
			<span>{error}</span>
		</div>
	{:else if job}
		<!-- Header -->
		<div class="mb-6">
			<div class="flex items-center justify-between">
				<div>
					<h1 class="text-4xl font-bold">{job.filename}</h1>
					<p class="text-base-content/70 mt-2">
						Job ID: <span class="font-mono text-sm">{jobId}</span>
					</p>
				</div>
				<div class="text-right">
					<div class="badge badge-lg {getStatusColor(job.status)}">
						{job.status.toUpperCase()}
					</div>
				</div>
			</div>
		</div>

		<!-- Status Card (if still processing) -->
		{#if job.status === 'running' || job.status === 'pending'}
			<div class="card bg-base-200 shadow-xl mb-6">
				<div class="card-body">
					<h2 class="card-title">
						<span class="loading loading-spinner text-primary"></span>
						Parsing in Progress
					</h2>
					<p>Your PCAP file is being processed by the GPU parser...</p>

					{#if job.total_packets > 0}
						<div class="mt-4">
							<div class="flex justify-between text-sm mb-1">
								<span>Packets parsed:</span>
								<span>{job.parsed_packets.toLocaleString()} / {job.total_packets.toLocaleString()}</span>
							</div>
							<progress
								class="progress progress-primary w-full"
								value={job.parsed_packets}
								max={job.total_packets}
							></progress>
						</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Summary Stats (if completed) -->
		{#if job.status === 'completed' && summary}
			<div class="stats stats-vertical lg:stats-horizontal shadow mb-6 w-full">
				<div class="stat">
					<div class="stat-figure text-primary">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							class="inline-block w-8 h-8 stroke-current"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
							></path>
						</svg>
					</div>
					<div class="stat-title">Total Packets</div>
					<div class="stat-value text-primary">{summary.total_packets.toLocaleString()}</div>
					<div class="stat-desc">Parsed successfully</div>
				</div>

				<div class="stat">
					<div class="stat-figure text-secondary">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							class="inline-block w-8 h-8 stroke-current"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M13 10V3L4 14h7v7l9-11h-7z"
							></path>
						</svg>
					</div>
					<div class="stat-title">Total Flows</div>
					<div class="stat-value text-secondary">{summary.total_flows.toLocaleString()}</div>
					<div class="stat-desc">Unique conversations</div>
				</div>

				<div class="stat">
					<div class="stat-figure text-accent">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							class="inline-block w-8 h-8 stroke-current"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M12 6v6m0 0v6m0-6h6m-6 0H6"
							></path>
						</svg>
					</div>
					<div class="stat-title">Parse Time</div>
					<div class="stat-value text-accent">{(summary.parse_time_ms / 1000).toFixed(2)}s</div>
					<div class="stat-desc">GPU accelerated</div>
				</div>
			</div>

			<!-- Tabs -->
			<div class="tabs tabs-boxed mb-6" >
				<button class="tab {activeTab === 'overview' ? 'tab-active' : ''}" on:click={() => (activeTab = 'overview')}>
					Overview
				</button>
				<button class="tab {activeTab === 'sessions' ? 'tab-active' : ''}" on:click={() => (activeTab = 'sessions')}>
					Sessions
				</button>
				<button class="tab {activeTab === 'topology' ? 'tab-active' : ''}" on:click={() => (activeTab = 'topology')}>
					Network Topology
				</button>
				<button class="tab {activeTab === 'stats' ? 'tab-active' : ''}" on:click={() => (activeTab = 'stats')}>
					Statistics
				</button>
			</div>

			<!-- Tab Content -->
			<div class="tab-content">
				{#if activeTab === 'overview'}
					<div class="grid grid-cols-1 gap-6">
						<StatsCharts {jobId} />
					</div>
				{:else if activeTab === 'sessions'}
					<SessionList {jobId} />
				{:else if activeTab === 'topology'}
					<NetworkGraph {jobId} />
				{:else if activeTab === 'stats'}
					<StatsCharts {jobId} detailed={true} />
				{/if}
			</div>
		{/if}
	{/if}
</div>
