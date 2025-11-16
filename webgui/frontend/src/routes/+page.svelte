<script>
	import { goto } from '$app/navigation';
	import { uploadPcap, streamJobStatus } from '$lib/api/client.js';

	let uploading = false;
	let uploadProgress = 0;
	let error = null;
	let dragActive = false;

	async function handleFileUpload(event) {
		const file = event.target.files?.[0];
		if (file) {
			await uploadFile(file);
		}
	}

	async function handleDrop(event) {
		event.preventDefault();
		dragActive = false;

		const file = event.dataTransfer.files?.[0];
		if (file) {
			await uploadFile(file);
		}
	}

	async function uploadFile(file) {
		// Validate file extension
		const validExtensions = ['.pcap', '.pcapng'];
		const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));

		if (!validExtensions.includes(ext)) {
			error = `Invalid file type. Please upload a .pcap or .pcapng file.`;
			return;
		}

		uploading = true;
		uploadProgress = 0;
		error = null;

		try {
			const job = await uploadPcap(file);

			// Redirect to dashboard with job ID
			goto(`/dashboard/${job.job_id}`);
		} catch (err) {
			error = err.message;
			uploading = false;
		}
	}

	function handleDragOver(event) {
		event.preventDefault();
		dragActive = true;
	}

	function handleDragLeave() {
		dragActive = false;
	}
</script>

<svelte:head>
	<title>PCAP Analyzer - Upload</title>
</svelte:head>

<div class="flex flex-col items-center justify-center min-h-[60vh]">
	<div class="card bg-base-200 shadow-xl w-full max-w-2xl">
		<div class="card-body">
			<h2 class="card-title text-3xl justify-center mb-4">
				<span class="text-primary">⚡</span> Upload PCAP File
			</h2>

			<p class="text-center text-base-content/70 mb-6">
				Upload your PCAP file for GPU-accelerated analysis. Supports .pcap and .pcapng formats up to 10GB.
			</p>

			<!-- Upload Zone -->
			<div
				class="border-2 border-dashed rounded-lg p-12 text-center transition-all
					{dragActive ? 'border-primary bg-primary/10' : 'border-base-300 hover:border-primary/50'}"
				on:drop={handleDrop}
				on:dragover={handleDragOver}
				on:dragleave={handleDragLeave}
				role="button"
				tabindex="0"
			>
				{#if uploading}
					<div class="flex flex-col items-center space-y-4">
						<span class="loading loading-spinner loading-lg text-primary"></span>
						<p class="text-lg">Uploading...</p>
					</div>
				{:else}
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-16 w-16 mx-auto mb-4 text-base-content/50"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
						/>
					</svg>

					<p class="text-lg mb-4">Drag & drop your PCAP file here</p>
					<p class="text-sm text-base-content/50 mb-6">or</p>

					<label for="file-upload" class="btn btn-primary btn-lg">
						Choose File
					</label>
					<input
						id="file-upload"
						type="file"
						accept=".pcap,.pcapng"
						class="hidden"
						on:change={handleFileUpload}
					/>
				{/if}
			</div>

			{#if error}
				<div class="alert alert-error mt-4">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="stroke-current shrink-0 h-6 w-6"
						fill="none"
						viewBox="0 0 24 24"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
						/>
					</svg>
					<span>{error}</span>
				</div>
			{/if}

			<!-- Info Section -->
			<div class="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
				<div class="stat bg-base-300 rounded-lg">
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
								d="M13 10V3L4 14h7v7l9-11h-7z"
							></path>
						</svg>
					</div>
					<div class="stat-title">GPU Accelerated</div>
					<div class="stat-value text-primary">10x+</div>
					<div class="stat-desc">faster than CPU</div>
				</div>

				<div class="stat bg-base-300 rounded-lg">
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
								d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
							></path>
						</svg>
					</div>
					<div class="stat-title">Max File Size</div>
					<div class="stat-value text-secondary">10GB</div>
					<div class="stat-desc">per upload</div>
				</div>

				<div class="stat bg-base-300 rounded-lg">
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
								d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
							></path>
						</svg>
					</div>
					<div class="stat-title">Rich Visuals</div>
					<div class="stat-value text-accent">5+</div>
					<div class="stat-desc">visualization types</div>
				</div>
			</div>
		</div>
	</div>
</div>
