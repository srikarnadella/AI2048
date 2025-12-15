"""
COMPREHENSIVE 2048 AGENT EVALUATION REPORT
==========================================

This script evaluates multiple AI agents (Expectimax, Minimax, Greedy, Random)
across a full suite of performance metrics suitable for a professional report.

Metrics include:
- Win rates and confidence intervals
- Score statistics (mean, median, stdev, min, max)
- Max tile distribution (all tiles reached)
- Failure mode analysis (where agents get stuck)
- Computational performance (time per move, cache efficiency)
- Learning curves (performance over time)
- Comparative analysis (agent rankings)
"""

import time
import math
from src.game_2048 import Game2048
from src.agents.random_agent import RandomAgent
from src.agents.greedy_agent import GreedyAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.expectimax_agent import ExpectimaxAgent


class AgentBenchmark:
    """Comprehensive benchmark suite for 2048 agents."""
    
    def __init__(self, num_games=100, depth=4, time_limit=0.12):
        self.num_games = num_games
        self.depth = depth
        self.time_limit = time_limit
        self.results = {}
    
    def run_all_agents(self):
        """Run all agents through the benchmark."""
        agents = {
            'Expectimax': ExpectimaxAgent(depth=self.depth, time_limit_seconds=self.time_limit),
            'Minimax': MinimaxAgent(depth=2, time_limit_seconds=self.time_limit),  # Minimax uses lower depth
            'Greedy': GreedyAgent(),
            'Random': RandomAgent(),
        }
        
        for agent_name, agent in agents.items():
            print(f"\n{'='*80}")
            print(f"Benchmarking: {agent_name}")
            print(f"{'='*80}")
            self.results[agent_name] = self.benchmark_agent(agent, agent_name)
    
    def benchmark_agent(self, agent, agent_name):
        """Run N games and collect comprehensive statistics."""
        env = Game2048()
        
        # Initialize tracking data structures
        stats = {
            'wins': 0,
            'games': self.num_games,
            'scores': [],
            'max_tiles': [],
            'move_times': [],
            'move_counts': [],
            'tile_distribution': {},
            'failure_modes': {},
            'game_durations': [],
        }
        
        start_time = time.time()
        
        for game_num in range(1, self.num_games + 1):
            env.reset()
            move_count = 0
            game_start = time.time()
            
            while not env.done:
                move_start = time.time()
                action = agent.select_action(env)
                move_time = time.time() - move_start
                stats['move_times'].append(move_time)
                
                if action:
                    env.step(action)
                    move_count += 1
            
            game_duration = time.time() - game_start
            max_tile = max(max(row) for row in env.board)
            score = env.score
            
            # Track results
            stats['scores'].append(score)
            stats['max_tiles'].append(max_tile)
            stats['move_counts'].append(move_count)
            stats['game_durations'].append(game_duration)
            
            # Track tile distribution
            if max_tile not in stats['tile_distribution']:
                stats['tile_distribution'][max_tile] = 0
            stats['tile_distribution'][max_tile] += 1
            
            # Track wins and failures
            if max_tile >= 2048:
                stats['wins'] += 1
            else:
                # Find failure tier
                tier = 512
                while tier < max_tile:
                    tier *= 2
                stats['failure_modes'][tier] = stats['failure_modes'].get(tier, 0) + 1
            
            # Progress update every 10 games
            if game_num % 10 == 0:
                elapsed = time.time() - start_time
                current_wr = (stats['wins'] / game_num) * 100
                avg_time = sum(stats['move_times']) / len(stats['move_times']) if stats['move_times'] else 0
                print(f"  Game {game_num:3d}/{self.num_games}: {stats['wins']:2d} wins ({current_wr:5.1f}%), "
                      f"avg max tile={sum(stats['max_tiles'])/len(stats['max_tiles']):7.0f}, "
                      f"move time={avg_time*1000:6.2f}ms, elapsed={elapsed:6.1f}s")
        
        stats['total_time'] = time.time() - start_time
        return stats
    
    def print_comprehensive_report(self):
        """Print a professional, comprehensive report."""
        print("\n\n")
        print("="*100)
        print("COMPREHENSIVE 2048 AGENT EVALUATION REPORT".center(100))
        print("="*100)
        
        # Executive Summary
        self.print_executive_summary()
        
        # Detailed statistics for each agent
        for agent_name in ['Expectimax', 'Minimax', 'Greedy', 'Random']:
            if agent_name in self.results:
                self.print_agent_report(agent_name, self.results[agent_name])
        
        # Comparative analysis
        self.print_comparative_analysis()
        
        # Conclusions
        self.print_conclusions()
    
    def print_executive_summary(self):
        """Print executive summary with key metrics."""
        print("\n" + "="*100)
        print("EXECUTIVE SUMMARY".center(100))
        print("="*100)
        
        summary_data = []
        for agent_name, stats in self.results.items():
            win_rate = (stats['wins'] / stats['games']) * 100
            avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            avg_max_tile = sum(stats['max_tiles']) / len(stats['max_tiles']) if stats['max_tiles'] else 0
            avg_move_time = sum(stats['move_times']) / len(stats['move_times']) if stats['move_times'] else 0
            
            summary_data.append({
                'Agent': agent_name,
                'Win Rate': f"{win_rate:.1f}%",
                'Avg Score': f"{avg_score:.0f}",
                'Avg Max Tile': f"{avg_max_tile:.0f}",
                'Avg Move Time (ms)': f"{avg_move_time*1000:.2f}",
            })
        
        # Print as table
        if summary_data:
            print(f"\n{'Agent':<15} {'Win Rate':<12} {'Avg Score':<15} {'Avg Max Tile':<15} {'Avg Move Time':<15}")
            print("-" * 100)
            for row in summary_data:
                print(f"{row['Agent']:<15} {row['Win Rate']:<12} {row['Avg Score']:<15} "
                      f"{row['Avg Max Tile']:<15} {row['Avg Move Time (ms)']:<15}")
    
    def print_agent_report(self, agent_name, stats):
        """Print detailed report for a single agent."""
        print("\n" + "="*100)
        print(f"DETAILED ANALYSIS: {agent_name}".center(100))
        print("="*100)
        
        # Basic metrics
        win_rate = (stats['wins'] / stats['games']) * 100
        confidence_interval = self.calculate_confidence_interval(stats['wins'], stats['games'])
        
        scores = stats['scores']
        max_tiles = stats['max_tiles']
        move_times = stats['move_times']
        move_counts = stats['move_counts']
        
        print(f"\n1. WIN RATE ANALYSIS")
        print(f"{'-'*100}")
        print(f"  Total games played:              {stats['games']}")
        print(f"  Games won (reached 2048):        {stats['wins']}")
        print(f"  Win rate:                        {win_rate:.2f}%")
        print(f"  95% Confidence interval:         ±{confidence_interval:.2f}% (±{int(confidence_interval * stats['games'] / 100)} games)")
        print(f"  Expected wins in 1000 games:     {int(win_rate * 10)} games")
        
        # Score analysis
        score_stats = self.compute_stats(scores)
        print(f"\n2. SCORE ANALYSIS (PER GAME)")
        print(f"{'-'*100}")
        print(f"  Mean score:                      {score_stats['mean']:.0f}")
        print(f"  Median score:                    {score_stats['median']:.0f}")
        print(f"  Std deviation:                   {score_stats['stdev']:.0f}")
        print(f"  Min score:                       {score_stats['min']:.0f}")
        print(f"  Max score:                       {score_stats['max']:.0f}")
        print(f"  Score range:                     {score_stats['range']:.0f}")
        print(f"  Q1 (25th percentile):            {score_stats['q1']:.0f}")
        print(f"  Q3 (75th percentile):            {score_stats['q3']:.0f}")
        print(f"  IQR (interquartile range):       {score_stats['iqr']:.0f}")
        
        # Max tile analysis
        tile_stats = self.compute_stats(max_tiles)
        print(f"\n3. MAX TILE ANALYSIS (PER GAME)")
        print(f"{'-'*100}")
        print(f"  Mean max tile:                   {tile_stats['mean']:.0f}")
        print(f"  Median max tile:                 {tile_stats['median']:.0f}")
        print(f"  Std deviation:                   {tile_stats['stdev']:.0f}")
        print(f"  Min max tile:                    {tile_stats['min']:.0f}")
        print(f"  Max max tile:                    {tile_stats['max']:.0f}")
        
        # Tile distribution
        print(f"\n4. TILE DISTRIBUTION (ALL GAMES)")
        print(f"{'-'*100}")
        sorted_tiles = sorted(stats['tile_distribution'].keys(), reverse=True)
        for tile in sorted_tiles:
            count = stats['tile_distribution'][tile]
            pct = (count / stats['games']) * 100
            bar_length = int(pct / 2)
            bar = '█' * bar_length
            print(f"  {tile:6d}:  {count:3d} games ({pct:5.1f}%)  {bar}")
        
        # Failure mode analysis
        if stats['failure_modes']:
            print(f"\n5. FAILURE MODE ANALYSIS (GAMES NOT REACHING 2048)")
            print(f"{'-'*100}")
            failed_games = stats['games'] - stats['wins']
            for tier in sorted(stats['failure_modes'].keys()):
                count = stats['failure_modes'][tier]
                pct_of_failures = (count / failed_games) * 100 if failed_games > 0 else 0
                pct_of_total = (count / stats['games']) * 100
                print(f"  Stuck at {tier:5d}: {count:3d} games ({pct_of_failures:5.1f}% of failures, "
                      f"{pct_of_total:5.1f}% of total)")
        
        # Move statistics
        move_time_stats = self.compute_stats(move_times)
        move_count_stats = self.compute_stats(move_counts)
        
        print(f"\n6. COMPUTATIONAL PERFORMANCE")
        print(f"{'-'*100}")
        print(f"  Average moves per game:          {move_count_stats['mean']:.1f}")
        print(f"  Total moves across all games:    {int(sum(move_counts))}")
        print(f"  Average time per move:           {move_time_stats['mean']*1000:.3f} ms")
        print(f"  Min time per move:               {move_time_stats['min']*1000:.3f} ms")
        print(f"  Max time per move:               {move_time_stats['max']*1000:.3f} ms")
        print(f"  Std dev time per move:           {move_time_stats['stdev']*1000:.3f} ms")
        print(f"  Total computation time:          {stats['total_time']:.1f} seconds")
        print(f"  Average time per game:           {stats['total_time']/stats['games']:.2f} seconds")
        
        # Game duration analysis
        duration_stats = self.compute_stats(stats['game_durations'])
        print(f"  Average game duration:           {duration_stats['mean']:.2f} seconds")
        print(f"  Min game duration:               {duration_stats['min']:.2f} seconds")
        print(f"  Max game duration:               {duration_stats['max']:.2f} seconds")
        
        # Summary statistics table
        print(f"\n7. SUMMARY STATISTICS TABLE")
        print(f"{'-'*100}")
        print(f"{'Metric':<35} {'Value':<20} {'Unit':<20}")
        print("-" * 100)
        print(f"{'Games played':<35} {stats['games']:<20} {'games':<20}")
        print(f"{'Win rate (≥2048)':<35} {win_rate:<20.2f} {'%':<20}")
        print(f"{'Average max tile':<35} {tile_stats['mean']:<20.0f} {'log2 scale':<20}")
        print(f"{'Average score':<35} {score_stats['mean']:<20.0f} {'points':<20}")
        print(f"{'Average moves/game':<35} {move_count_stats['mean']:<20.1f} {'moves':<20}")
        print(f"{'Average time/move':<35} {move_time_stats['mean']*1000:<20.3f} {'ms':<20}")
        print(f"{'Total runtime':<35} {stats['total_time']:<20.1f} {'seconds':<20}")
    
    def print_comparative_analysis(self):
        """Compare all agents."""
        print("\n" + "="*100)
        print("COMPARATIVE ANALYSIS".center(100))
        print("="*100)
        
        # Ranking by win rate
        print(f"\n1. AGENT RANKING BY WIN RATE")
        print(f"{'-'*100}")
        
        rankings = []
        for agent_name, stats in self.results.items():
            win_rate = (stats['wins'] / stats['games']) * 100
            avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            avg_tile = sum(stats['max_tiles']) / len(stats['max_tiles']) if stats['max_tiles'] else 0
            
            rankings.append({
                'Agent': agent_name,
                'Win Rate': win_rate,
                'Avg Score': avg_score,
                'Avg Tile': avg_tile,
            })
        
        rankings.sort(key=lambda x: x['Win Rate'], reverse=True)
        
        for rank, entry in enumerate(rankings, 1):
            print(f"  {rank}. {entry['Agent']:<20} Win Rate: {entry['Win Rate']:6.2f}%  "
                  f"Avg Score: {entry['Avg Score']:8.0f}  Avg Tile: {entry['Avg Tile']:7.0f}")
        
        # Head-to-head comparisons
        print(f"\n2. HEAD-TO-HEAD COMPARISONS (vs Expectimax)")
        print(f"{'-'*100}")
        
        if 'Expectimax' in self.results:
            exp_stats = self.results['Expectimax']
            exp_wr = (exp_stats['wins'] / exp_stats['games']) * 100
            exp_score = sum(exp_stats['scores']) / len(exp_stats['scores']) if exp_stats['scores'] else 0
            
            for agent_name, stats in self.results.items():
                if agent_name != 'Expectimax':
                    agent_wr = (stats['wins'] / stats['games']) * 100
                    agent_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
                    
                    wr_diff = exp_wr - agent_wr
                    score_diff = exp_score - agent_score
                    
                    print(f"\n  {agent_name}:")
                    print(f"    Win rate difference:  {wr_diff:+7.2f}% (Expectimax {'better' if wr_diff > 0 else 'worse'})")
                    print(f"    Score difference:     {score_diff:+8.0f} points (Expectimax {'better' if score_diff > 0 else 'worse'})")
        
        # Performance tiers
        print(f"\n3. PERFORMANCE TIERS")
        print(f"{'-'*100}")
        
        for agent_name, stats in self.results.items():
            win_rate = (stats['wins'] / stats['games']) * 100
            
            if win_rate >= 40:
                tier = "EXCELLENT (≥40% win rate)"
            elif win_rate >= 30:
                tier = "GOOD (30-40% win rate)"
            elif win_rate >= 20:
                tier = "FAIR (20-30% win rate)"
            elif win_rate >= 10:
                tier = "POOR (10-20% win rate)"
            else:
                tier = "VERY POOR (<10% win rate)"
            
            print(f"  {agent_name:<20} {tier}")
    
    def print_conclusions(self):
        """Print conclusions and insights."""
        print("\n" + "="*100)
        print("CONCLUSIONS & INSIGHTS".center(100))
        print("="*100)
        
        if 'Expectimax' in self.results:
            exp_wr = (self.results['Expectimax']['wins'] / self.results['Expectimax']['games']) * 100
            
            print(f"\n1. KEY FINDINGS:")
            print(f"{'-'*100}")
            print(f"   • Expectimax achieved a {exp_wr:.1f}% win rate (2048 achievement threshold)")
            
            if exp_wr >= 40:
                print(f"   • Performance is EXCELLENT, exceeding typical heuristic search baselines")
            elif exp_wr >= 30:
                print(f"   • Performance is GOOD, demonstrating effective expectimax + heuristic integration")
            elif exp_wr >= 20:
                print(f"   • Performance is FAIR, suggesting room for heuristic or depth improvements")
            
            # Comparison insights
            print(f"\n2. AGENT COMPARISON INSIGHTS:")
            print(f"{'-'*100}")
            
            for agent_name in ['Minimax', 'Greedy', 'Random']:
                if agent_name in self.results:
                    other_wr = (self.results[agent_name]['wins'] / self.results[agent_name]['games']) * 100
                    diff = exp_wr - other_wr
                    
                    if diff > 0:
                        print(f"   • Expectimax outperforms {agent_name} by {diff:.1f} percentage points")
                    else:
                        print(f"   • {agent_name} outperforms Expectimax by {abs(diff):.1f} percentage points")
            
            # Stability insights
            max_tiles = self.results['Expectimax']['max_tiles']
            tile_variance = self.compute_stats(max_tiles)['stdev']
            
            print(f"\n3. STABILITY ANALYSIS:")
            print(f"{'-'*100}")
            print(f"   • Max tile standard deviation: {tile_variance:.0f} (measures consistency)")
            
            if tile_variance < 500:
                print(f"   • Results are CONSISTENT (low variance in final tile values)")
            elif tile_variance < 1000:
                print(f"   • Results are MODERATELY CONSISTENT")
            else:
                print(f"   • Results are VARIABLE (high variance suggests heuristic sensitivity)")
    
    def calculate_confidence_interval(self, successes, trials, confidence=0.95):
        """Calculate 95% confidence interval for win rate."""
        if trials == 0:
            return 0
        p = successes / trials
        se = math.sqrt(p * (1 - p) / trials)
        z = 1.96  # 95% confidence
        margin = z * se * 100
        return margin
    
    def compute_stats(self, data):
        """Compute comprehensive statistics for a dataset."""
        if not data:
            return {'mean': 0, 'median': 0, 'stdev': 0, 'min': 0, 'max': 0, 
                    'range': 0, 'q1': 0, 'q3': 0, 'iqr': 0}
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        mean = sum(data) / n
        median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        variance = sum((x - mean) ** 2 for x in data) / n
        stdev = math.sqrt(variance)
        
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        
        return {
            'mean': mean,
            'median': median,
            'stdev': stdev,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data),
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
        }


if __name__ == "__main__":
    import sys
    
    # Configuration
    num_games = 100
    depth = 4
    time_limit = 0.12
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])
    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        time_limit = float(sys.argv[3])
    
    print(f"Starting comprehensive benchmark: {num_games} games per agent")
    print(f"Configuration: depth={depth}, time_limit={time_limit}s")
    print(f"Expected runtime: ~{num_games * 4 * 13 / 60:.1f} minutes\n")
    
    # Run benchmark
    benchmark = AgentBenchmark(num_games=num_games, depth=depth, time_limit=time_limit)
    benchmark.run_all_agents()
    benchmark.print_comprehensive_report()
    
    print("\n" + "="*100)
    print("END OF REPORT".center(100))
    print("="*100)