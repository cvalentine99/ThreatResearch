<script>
	import { onMount } from 'svelte';
	import { getProtocolDistribution } from '$lib/api/client.js';
	import * as echarts from 'echarts';

	export let jobId;
	export let detailed = false;

	let protocolChartContainer;
	let protocolChart;
	let loading = true;
	let error = null;

	onMount(async () => {
		await loadCharts();
	});

	async function loadCharts() {
		try {
			console.log('StatsCharts: Loading data for job', jobId);
			const protocolData = await getProtocolDistribution(jobId);
			console.log('StatsCharts: Got protocol data', protocolData);

			// Initialize protocol distribution chart
			if (protocolChartContainer) {
				protocolChart = echarts.init(protocolChartContainer, 'dark');

				const chartData = protocolData.protocols.map((p) => ({
					name: p.protocol,
					value: p.count
				}));

				const option = {
					title: {
						text: 'Protocol Distribution',
						left: 'center',
						textStyle: { color: '#fff' }
					},
					tooltip: {
						trigger: 'item',
						formatter: '{b}: {c} ({d}%)'
					},
					legend: {
						orient: 'vertical',
						right: 10,
						top: 'center',
						textStyle: { color: '#fff' }
					},
					series: [
						{
							name: 'Protocols',
							type: 'pie',
							radius: ['40%', '70%'],
							avoidLabelOverlap: false,
							itemStyle: {
								borderRadius: 10,
								borderColor: '#0f172a',
								borderWidth: 2
							},
							label: {
								show: true,
								formatter: '{b}\n{d}%',
								color: '#fff'
							},
							emphasis: {
								label: {
									show: true,
									fontSize: 16,
									fontWeight: 'bold'
								}
							},
							data: chartData,
							colorBy: 'data'
						}
					],
					color: ['#3b82f6', '#8b5cf6', '#22d3ee', '#f59e0b', '#10b981', '#ef4444']
				};

				protocolChart.setOption(option);

				// Handle window resize
				window.addEventListener('resize', () => {
					if (protocolChart) protocolChart.resize();
				});
			}

			loading = false;
		} catch (err) {
			error = err.message;
			loading = false;
		}
	}
</script>

<div class="grid grid-cols-1 {detailed ? 'lg:grid-cols-2' : ''} gap-6">
	<div class="card bg-base-200 shadow-xl">
		<div class="card-body">
			{#if loading}
				<div class="flex justify-center items-center h-[400px]">
					<span class="loading loading-spinner loading-lg text-primary"></span>
				</div>
			{:else if error}
				<div class="alert alert-error">
					<span>Error loading charts: {error}</span>
				</div>
			{:else}
				<div bind:this={protocolChartContainer} class="w-full h-[400px]"></div>
			{/if}
		</div>
	</div>

	{#if detailed}
		<div class="card bg-base-200 shadow-xl">
			<div class="card-body">
				<h2 class="card-title mb-4">Additional Statistics</h2>
				<p class="text-base-content/70">
					More detailed visualizations coming soon:
				</p>
				<ul class="list-disc list-inside space-y-2 mt-4">
					<li>Bandwidth over time (timeline chart)</li>
					<li>Top talkers (bar chart)</li>
					<li>Traffic heatmap</li>
					<li>Geo-IP mapping</li>
				</ul>
			</div>
		</div>
	{/if}
</div>

<style>
	:global(.echarts-container) {
		width: 100%;
		height: 100%;
	}
</style>
