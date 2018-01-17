#!/usr/bin/env ruby

require 'csv'
require 'matplotlib/pyplot'

times = CSV.parse(File.read('ex1.csv'))

0.upto(times.size - 1) do |i|
  times[i] = times[i].map { |x| x.to_f}
end

nrm2_gflop = times[0].map { |size|   4e-9 * size }
gemv_gflop = times[0].map { |size|   7e-9 * size * size }
gemm_gflop = times[0].map { |size|  10e-9 * size * size * size }

nrm2_perf = []
gemv_perf = []
gemm_perf = []

nrm2_gflop.each_with_index do |gflop, idx|
  nrm2_perf << gflop / times[1][idx]
end

gemv_gflop.each_with_index do |gflop, idx|
  gemv_perf << gflop / times[2][idx]
end

gemm_gflop.each_with_index do |gflop, idx|
  gemm_perf << gflop / times[3][idx]
end

plt = Matplotlib::Pyplot

plt.xlabel("Problem Sizes")
plt.ylabel("GFlop/s")

plt.plot(times[0], nrm2_perf, 'r', label='nrm2')
plt.plot(times[0], gemv_perf, 'g', label='gemv')
plt.plot(times[0], gemm_perf, 'b', label='gemm')

plt.legend()

plt.savefig('ex1.png')
