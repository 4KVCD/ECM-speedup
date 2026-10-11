// Measurement probe, not part of the encoder: grain of the candidate loops of the mode search, printed at exit.
// insert_grain_probes.py copies this file to source/Lib/CommonLib of a throwaway tree and adds the probe sites.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>

// Number of insertions into ranked candidate lists (updateCandList, orderCandList); counts the candidates a phase
// evaluates when the per-candidate work is not timed separately.
inline long long g_probeCandInserts = 0;

struct ProbeGrain
{
  static constexpr int    numPools      = 4;
  static constexpr int    poolSize[numPools] = { 2, 4, 8, 16 };
  static constexpr double taskOverhead  = 2e-6;   // dispatch and join cost per task on a spinning pool

  struct Entry
  {
    long long loops = 0, tasks = 0;
    double    sum = 0;
    double    parallel[numPools] = {};
    double    sumByTaskSize[5] = {};   // task time < 5 us, < 20 us, < 100 us, < 1 ms, larger
    double    sumByLoopSize[4] = {};   // loops with 1, 2-3, 4-7, 8+ tasks
  };
  std::map<std::string, Entry> entries;

  static ProbeGrain& get()
  {
    static ProbeGrain g;
    return g;
  }

  // Records one loop instance: n tasks, their total and longest time; uniform = tasks of equal length assumed.
  void record( const std::string& name, int n, double sum, double maxTask )
  {
    Entry& e = entries[name];
    e.loops++;
    e.tasks += n;
    e.sum += sum;
    for( int p = 0; p < numPools; p++ )
    {
      const int waves = ( n + poolSize[p] - 1 ) / poolSize[p];
      e.parallel[p] += std::max( maxTask, sum / poolSize[p] ) + taskOverhead * waves;
    }
    const double avg = sum / n;
    e.sumByTaskSize[avg < 5e-6 ? 0 : avg < 20e-6 ? 1 : avg < 100e-6 ? 2 : avg < 1e-3 ? 3 : 4] += sum;
    e.sumByLoopSize[n == 1 ? 0 : n < 4 ? 1 : n < 8 ? 2 : 3] += sum;
  }

  ~ProbeGrain()
  {
    fprintf( stderr, "\n[grain] name: loops, tasks, tasks/loop, total s | speedup with 2/4/8/16 workers | "
                     "time share by task size <5us <20us <100us <1ms >1ms | by loop size 1, 2-3, 4-7, 8+\n" );
    for( const auto& [name, e]: entries )
    {
      fprintf( stderr, "[grain] %-22s %8lld %10lld %6.1f %9.2f |", name.c_str(), e.loops, e.tasks,
               double( e.tasks ) / e.loops, e.sum );
      for( int p = 0; p < numPools; p++ )
      {
        fprintf( stderr, " %5.2f", e.sum / e.parallel[p] );
      }
      fprintf( stderr, " |" );
      for( double s: e.sumByTaskSize )
      {
        fprintf( stderr, " %4.0f%%", 100 * s / e.sum );
      }
      fprintf( stderr, " |" );
      for( double s: e.sumByLoopSize )
      {
        fprintf( stderr, " %4.0f%%", 100 * s / e.sum );
      }
      fprintf( stderr, "\n" );
    }
  }
};

// One instance of a candidate loop; tasks add their times, the destructor records the instance.
struct ProbeLoop
{
  std::string name;
  int         n       = 0;
  double      sum     = 0;
  double      maxTask = 0;

  explicit ProbeLoop( std::string nm ) : name( std::move( nm ) ) {}
  void add( double t )
  {
    n++;
    sum += t;
    maxTask = std::max( maxTask, t );
  }
  ~ProbeLoop()
  {
    if( n )
    {
      ProbeGrain::get().record( name, n, sum, maxTask );
    }
  }
};

// One loop iteration; the selector picks the loop it belongs to when it ends (e.g. ISP or not).
template<class Select> struct ProbeTask
{
  Select                                select;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  explicit ProbeTask( Select s ) : select( s ) {}
  ~ProbeTask() { select().add( std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count() ); }
};
template<class Select> ProbeTask<Select> makeProbeTask( Select s ) { return ProbeTask<Select>( s ); }

// A phase whose candidates are not timed one by one: their number is the count of candidate-list insertions
// and equal task lengths are assumed.
struct ProbePhaseCount
{
  std::string                           name;
  long long                             inserts0 = g_probeCandInserts;
  std::chrono::steady_clock::time_point start    = std::chrono::steady_clock::now();
  bool                                  open     = true;

  explicit ProbePhaseCount( std::string nm ) : name( std::move( nm ) ) {}
  void finish()
  {
    if( !open )
    {
      return;
    }
    open                = false;
    const double t      = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
    const int    n      = int( std::max( 1LL, g_probeCandInserts - inserts0 ) );
    ProbeGrain::get().record( name, n, t, t / n );
  }
  ~ProbePhaseCount() { finish(); }
};
