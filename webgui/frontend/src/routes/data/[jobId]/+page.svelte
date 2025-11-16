<script>
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { getJobSummary, getProtocolDistribution, getFlows } from '$lib/api/client.js';

	$: jobId = $page.params.jobId;

	let summary = null;
	let protocols = null;
	let flows = null;
	let loading = true;

	onMount(async () => {
		try {
			summary = await getJobSummary(jobId);
			protocols = await getProtocolDistribution(jobId);
			flows = await getFlows(jobId, { limit: 100 });
			loading = false;
		} catch (err) {
			console.error('Error loading data:', err);
			loading = false;
		}
	});
</script>

<svelte:head>
	<title>Raw Data - {jobId}</title>
</svelte:head>

<div style="padding: 2rem; background: #1a1a1a; color: #fff; min-height: 100vh;">
	<h1 style="font-size: 2rem; margin-bottom: 2rem;">RAW DATA VIEW</h1>

	{#if loading}
		<p style="font-size: 1.5rem;">Loading data...</p>
	{:else}
		<!-- Summary -->
		<div style="background: #2a2a2a; padding: 1.5rem; margin-bottom: 2rem; border: 2px solid #00ff00;">
			<h2 style="color: #00ff00; font-size: 1.5rem; margin-bottom: 1rem;">JOB SUMMARY</h2>
			{#if summary}
				<pre style="color: #fff; font-size: 1.1rem; line-height: 1.8;">{JSON.stringify(summary, null, 2)}</pre>
			{:else}
				<p style="color: red;">NO SUMMARY DATA</p>
			{/if}
		</div>

		<!-- Protocols -->
		<div style="background: #2a2a2a; padding: 1.5rem; margin-bottom: 2rem; border: 2px solid #00ff00;">
			<h2 style="color: #00ff00; font-size: 1.5rem; margin-bottom: 1rem;">PROTOCOL DISTRIBUTION</h2>
			{#if protocols && protocols.protocols}
				<table style="width: 100%; color: #fff; font-size: 1.1rem;">
					<thead>
						<tr style="background: #333;">
							<th style="padding: 1rem; text-align: left; border: 1px solid #555;">Protocol</th>
							<th style="padding: 1rem; text-align: left; border: 1px solid #555;">Packet Count</th>
							<th style="padding: 1rem; text-align: left; border: 1px solid #555;">Percentage</th>
						</tr>
					</thead>
					<tbody>
						{#each protocols.protocols as proto}
							<tr style="background: #2a2a2a;">
								<td style="padding: 1rem; border: 1px solid #555; font-weight: bold; color: #00ff00;">{proto.protocol}</td>
								<td style="padding: 1rem; border: 1px solid #555;">{proto.count.toLocaleString()}</td>
								<td style="padding: 1rem; border: 1px solid #555;">
									{((proto.count / summary.total_packets) * 100).toFixed(2)}%
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{:else}
				<p style="color: red;">NO PROTOCOL DATA</p>
			{/if}
		</div>

		<!-- Flows -->
		<div style="background: #2a2a2a; padding: 1.5rem; margin-bottom: 2rem; border: 2px solid #00ff00;">
			<h2 style="color: #00ff00; font-size: 1.5rem; margin-bottom: 1rem;">TOP 100 FLOWS</h2>
			{#if flows && flows.flows}
				<p style="color: #0ff; margin-bottom: 1rem; font-size: 1.1rem;">
					Total Flows: {flows.total.toLocaleString()} | Showing: {flows.flows.length}
				</p>
				<div style="overflow-x: auto;">
					<table style="width: 100%; color: #fff; font-size: 0.95rem;">
						<thead>
							<tr style="background: #333;">
								<th style="padding: 0.75rem; text-align: left; border: 1px solid #555;">Source IP</th>
								<th style="padding: 0.75rem; text-align: left; border: 1px solid #555;">Port</th>
								<th style="padding: 0.75rem; text-align: left; border: 1px solid #555;">Dest IP</th>
								<th style="padding: 0.75rem; text-align: left; border: 1px solid #555;">Port</th>
								<th style="padding: 0.75rem; text-align: left; border: 1px solid #555;">Protocol</th>
								<th style="padding: 0.75rem; text-align: left; border: 1px solid #555;">Packets</th>
							</tr>
						</thead>
						<tbody>
							{#each flows.flows as flow}
								<tr style="background: #2a2a2a;">
									<td style="padding: 0.75rem; border: 1px solid #555; font-family: monospace; color: #0ff;">{flow.src_ip}</td>
									<td style="padding: 0.75rem; border: 1px solid #555;">{flow.src_port}</td>
									<td style="padding: 0.75rem; border: 1px solid #555; font-family: monospace; color: #0ff;">{flow.dst_ip}</td>
									<td style="padding: 0.75rem; border: 1px solid #555;">{flow.dst_port}</td>
									<td style="padding: 0.75rem; border: 1px solid #555;">
										<span style="background: #555; padding: 0.25rem 0.5rem; border-radius: 4px;">{flow.protocol}</span>
									</td>
									<td style="padding: 0.75rem; border: 1px solid #555; font-weight: bold; color: #0f0;">
										{flow.packet_count.toLocaleString()}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<p style="color: red;">NO FLOW DATA</p>
			{/if}
		</div>
	{/if}
</div>
