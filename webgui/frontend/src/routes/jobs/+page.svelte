<script>
	import { onMount } from 'svelte';
	import { listJobs } from '$lib/api/client.js';
	import { goto } from '$app/navigation';

	let jobs = [];
	let loading = true;
	let error = null;

	onMount(async () => {
		try {
			jobs = await listJobs();
		} catch (err) {
			error = err.message;
		} finally {
			loading = false;
		}
	});

	function getStatusBadge(status) {
		const badges = {
			pending: 'badge-warning',
			running: 'badge-info',
			completed: 'badge-success',
			failed: 'badge-error'
		};
		return badges[status] || 'badge-ghost';
	}

	function formatFileSize(bytes) {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
	}

	function formatDate(dateString) {
		return new Date(dateString).toLocaleString();
	}

	function viewJob(jobId) {
		goto(`/dashboard/${jobId}`);
	}
</script>

<svelte:head>
	<title>PCAP Analyzer - Jobs</title>
</svelte:head>

<div class="animate-fade-in">
	<h1 class="text-4xl font-bold mb-6">Parsing Jobs</h1>

	{#if loading}
		<div class="flex justify-center items-center min-h-[400px]">
			<span class="loading loading-spinner loading-lg text-primary"></span>
		</div>
	{:else if error}
		<div class="alert alert-error">
			<span>Error loading jobs: {error}</span>
		</div>
	{:else if jobs.length === 0}
		<div class="card bg-base-200">
			<div class="card-body items-center text-center">
				<h2 class="card-title">No jobs yet</h2>
				<p>Upload a PCAP file to get started</p>
				<div class="card-actions">
					<a href="/" class="btn btn-primary">Upload File</a>
				</div>
			</div>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="table table-zebra">
				<thead>
					<tr>
						<th>Filename</th>
						<th>Size</th>
						<th>Status</th>
						<th>Packets</th>
						<th>Created</th>
						<th>Actions</th>
					</tr>
				</thead>
				<tbody>
					{#each jobs as job}
						<tr class="hover">
							<td class="font-mono text-sm">{job.filename}</td>
							<td>{formatFileSize(job.file_size)}</td>
							<td>
								<span class="badge {getStatusBadge(job.status)}">
									{job.status}
								</span>
							</td>
							<td>{job.parsed_packets.toLocaleString()}</td>
							<td class="text-sm">{formatDate(job.created_at)}</td>
							<td>
								{#if job.status === 'completed'}
									<button class="btn btn-sm btn-primary" on:click={() => viewJob(job.job_id)}>
										View Dashboard
									</button>
								{:else if job.status === 'running'}
									<button class="btn btn-sm btn-ghost" on:click={() => viewJob(job.job_id)}>
										View Status
									</button>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
