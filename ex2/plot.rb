#!/usr/bin/env ruby

require 'csv'
require 'matplotlib/pyplot'

times = CSV.parse(File.read('ex2.csv'))

0.upto(times.size - 1) do |i|
  times[i] = times[i].map { |x| x.to_f}
end

getrf_n_gflop = times[0].map { |size| (2.0 / 3.0) * size**3 }
getrf_p_gflop = times[0].map { |size| (2.0 / 3.0) * size**3 }

getrf_n_perf = []
getrf_p_perf = []

getrf_n_gflop.each_with_index do |gflop, idx|
  getrf_n_perf << gflop / times[1][idx]
end

getrf_p_gflop.each_with_index do |gflop, idx|
  getrf_p_perf << gflop / times[2][idx]
end

plt = Matplotlib::Pyplot

plt.xlabel("Problem Sizes")
plt.ylabel("GFlop/s")

line1 = plt.plot(times[0], getrf_n_perf, 'g')
line2 = plt.plot(times[0], getrf_p_perf, 'b')

plt.savefig('ex2.png')
