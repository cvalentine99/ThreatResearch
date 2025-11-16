<script>
	import '../app.css';
</script>

<div class="min-h-screen bg-base-100">
	<!-- Navigation -->
	<div class="navbar bg-base-200 shadow-lg">
		<div class="flex-1">
			<a href="/" class="btn btn-ghost text-xl">
				<span class="text-primary">⚡</span> PCAP Analyzer
			</a>
		</div>
		<div class="flex-none">
			<ul class="menu menu-horizontal px-1">
				<li><a href="/">Upload</a></li>
				<li><a href="/jobs">Jobs</a></li>
			</ul>
		</div>
	</div>

	<!-- Main content -->
	<main class="container mx-auto p-4">
		<slot />
	</main>

	<!-- Footer -->
	<footer class="footer footer-center p-4 bg-base-200 text-base-content mt-8">
		<div>
			<p>GPU-Accelerated PCAP Analysis | v0.1.0</p>
		</div>
	</footer>
</div>
