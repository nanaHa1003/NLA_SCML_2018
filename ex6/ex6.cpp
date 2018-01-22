#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // 16KB upto 25MB
  std::vector<int>    test_sizes = {{ 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 25600 }};
  std::vector<double> save_times(test_sizes.size());
  std::vector<double> elap_times(test_sizes.size());

  for (size_t i = 0; i < test_sizes.size(); ++i) {
    std::vector<char> packet(test_sizes[i] * 1024);
    for (int j = 0; j < 100; ++j) {
      std::cout << ".";
      double tp;
      if (mpi_rank == 0) {
        tp = MPI_Wtime();
        MPI_Send(static_cast<void *>(packet.data()), packet.size(), MPI_BYTE, 1, test_sizes[i] + i, MPI_COMM_WORLD);
      } else if (mpi_rank == 1) {
        MPI_Status stat;
        MPI_Recv(static_cast<void *>(packet.data()), packet.size(), MPI_BYTE, 0, test_sizes[i] + i, MPI_COMM_WORLD, &stat);
        tp = MPI_Wtime();
      }
      save_times[i] += tp / 100.0;
    }
  }
  std::cout << std::endl;

  if (mpi_rank == 0) {
    MPI_Status stat;
    MPI_Recv(static_cast<void *>(elap_times.data()), elap_times.size(), MPI_DOUBLE, 1, 65536, MPI_COMM_WORLD, &stat);
  } else if (mpi_rank == 1) {
    MPI_Send(static_cast<void *>(save_times.data()), save_times.size(), MPI_DOUBLE, 0, 65536, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (mpi_rank == 0) {
    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
      std::cout << test_sizes[i] << ",";
    }
    std::cout << test_sizes.back() << std::endl;

    for (size_t i = 0; i < elap_times.size() - 1; ++i) {
      elap_times[i] -= save_times[i];
      std::cout << 1e-6 * test_sizes[i] / elap_times[i] << ",";
    }
    std::cout << 1e-6 * test_sizes.back() / (elap_times.back() - save_times.back()) << std::endl;
  }

  MPI_Finalize();
  return 0;
}
