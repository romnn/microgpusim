#include "stream_manager.hpp"

#include "trace_gpgpu_sim.hpp"
#include "trace_kernel_info.hpp"

#include <unistd.h>

stream_manager::stream_manager(trace_gpgpu_sim *gpu,
                               bool cuda_launch_blocking) {
  m_gpu = gpu;
  m_service_stream_zero = false;
  m_cuda_launch_blocking = cuda_launch_blocking;
  pthread_mutex_init(&m_lock, NULL);
  m_last_stream = m_streams.begin();
}

bool stream_manager::operation(bool *sim) {
  bool check = check_finished_kernel();
  pthread_mutex_lock(&m_lock);
  //    if(check)m_gpu->print_stats();
  stream_operation op = front();
  if (!op.do_operation(m_gpu))  // not ready to execute
  {
    // cancel operation
    if (op.is_kernel()) {
      unsigned grid_uid = op.get_kernel()->get_uid();
      m_grid_id_to_stream.erase(grid_uid);
    }
    op.get_stream()->cancel_front();
  }
  pthread_mutex_unlock(&m_lock);
  // pthread_mutex_lock(&m_lock);
  // simulate a clock cycle on the GPU
  return check;
}

bool stream_manager::check_finished_kernel() {
  unsigned grid_uid = m_gpu->finished_kernel();
  bool check = register_finished_kernel(grid_uid);
  return check;
}

bool stream_manager::register_finished_kernel(unsigned grid_uid) {
  // called by gpu simulation thread
  if (grid_uid > 0) {
    // ROMAN: check if stream exists
    std::map<unsigned int, CUstream_st *>::const_iterator it =
        m_grid_id_to_stream.find(grid_uid);
    if (it == m_grid_id_to_stream.end()) {
      // no stream present
      return false;
    }
    CUstream_st *stream = m_grid_id_to_stream[grid_uid];
    assert(stream != NULL);
    // ROMAN: check if stream is empty here
    if (stream->empty()) {
      return false;
    }
    trace_kernel_info_t *kernel = stream->front().get_kernel();
    assert(grid_uid == kernel->get_uid());

    // Jin: should check children kernels for CDP
    if (kernel->is_finished()) {
      //            std::ofstream kernel_stat("kernel_stat.txt",
      //            std::ofstream::out | std::ofstream::app); kernel_stat<< "
      //            kernel " << grid_uid << ": " << kernel->name();
      //            if(kernel->get_parent())
      //                kernel_stat << ", parent " <<
      //                kernel->get_parent()->get_uid() <<
      //                ", launch " << kernel->launch_cycle;
      //            kernel_stat<< ", start " << kernel->start_cycle <<
      //                ", end " << kernel->end_cycle << ", retire " <<
      //                gpu_sim_cycle + gpu_tot_sim_cycle << "\n";
      //            printf("kernel %d finishes, retires from stream %d\n",
      //            grid_uid, stream->get_uid()); kernel_stat.flush();
      //            kernel_stat.close();
      stream->record_next_done();
      m_grid_id_to_stream.erase(grid_uid);
      kernel->notify_parent_finished();
      delete kernel;
      return true;
    }
  }

  return false;
}

void stream_manager::stop_all_running_kernels() {
  pthread_mutex_lock(&m_lock);

  // Signal m_gpu to stop all running kernels
  m_gpu->stop_all_running_kernels();

  // Clean up all streams waiting on running kernels
  int count = 0;
  while (check_finished_kernel()) {
    count++;
  }

  // If any kernels completed, print out the current stats
  if (count > 0) m_gpu->print_stats();

  pthread_mutex_unlock(&m_lock);
}

stream_operation stream_manager::front() {
  // called by gpu simulation thread
  stream_operation result;
  //    if( concurrent_streams_empty() )
  m_service_stream_zero = true;
  if (m_service_stream_zero) {
    if (!m_stream_zero.empty() && !m_stream_zero.busy()) {
      result = m_stream_zero.next();
      if (result.is_kernel()) {
        unsigned grid_id = result.get_kernel()->get_uid();
        m_grid_id_to_stream[grid_id] = &m_stream_zero;
      }
    } else {
      m_service_stream_zero = false;
    }
  }
  if (!m_service_stream_zero) {
    std::list<struct CUstream_st *>::iterator s = m_last_stream;
    if (m_last_stream == m_streams.end()) {
      s = m_streams.begin();
    } else {
      s++;
    }
    for (size_t ii = 0; ii < m_streams.size(); ii++, s++) {
      if (s == m_streams.end()) {
        s = m_streams.begin();
      }
      m_last_stream = s;
      CUstream_st *stream = *s;
      if (!stream->busy() && !stream->empty()) {
        result = stream->next();
        if (result.is_kernel()) {
          unsigned grid_id = result.get_kernel()->get_uid();
          m_grid_id_to_stream[grid_id] = stream;
        }
        break;
      }
    }
  }
  return result;
}

void stream_manager::add_stream(struct CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  m_streams.push_back(stream);
  pthread_mutex_unlock(&m_lock);
}

void stream_manager::destroy_stream(CUstream_st *stream) {
  // called by host thread
  pthread_mutex_lock(&m_lock);
  while (!stream->empty())
    ;
  std::list<CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    if (*s == stream) {
      m_streams.erase(s);
      break;
    }
  }
  delete stream;
  m_last_stream = m_streams.begin();
  pthread_mutex_unlock(&m_lock);
}

bool stream_manager::concurrent_streams_empty() {
  bool result = true;
  if (m_streams.empty()) return true;
  // called by gpu simulation thread
  std::list<struct CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    struct CUstream_st *stream = *s;
    if (!stream->empty()) {
      // stream->print(stdout);
      result = false;
      break;
    }
  }
  return result;
}

bool stream_manager::empty_protected() {
  bool result = true;
  pthread_mutex_lock(&m_lock);
  if (!concurrent_streams_empty()) result = false;
  if (!m_stream_zero.empty()) result = false;
  pthread_mutex_unlock(&m_lock);
  return result;
}

bool stream_manager::empty() {
  bool result = true;
  if (!concurrent_streams_empty()) result = false;
  if (!m_stream_zero.empty()) result = false;
  return result;
}

void stream_manager::print(FILE *fp) {
  pthread_mutex_lock(&m_lock);
  print_impl(fp);
  pthread_mutex_unlock(&m_lock);
}
void stream_manager::print_impl(FILE *fp) {
  fprintf(fp, "GPGPU-Sim API: Stream Manager State\n");
  std::list<struct CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); ++s) {
    struct CUstream_st *stream = *s;
    if (!stream->empty()) stream->print(fp);
  }
  if (!m_stream_zero.empty()) m_stream_zero.print(fp);
}

void stream_manager::push(stream_operation op) {
  struct CUstream_st *stream = op.get_stream();

  // block if stream 0 (or concurrency disabled) and pending concurrent
  // operations exist
  bool block = !stream || m_cuda_launch_blocking;
  while (block) {
    pthread_mutex_lock(&m_lock);
    block = !concurrent_streams_empty();
    pthread_mutex_unlock(&m_lock);
  };

  pthread_mutex_lock(&m_lock);
  if (!m_gpu->cycle_insn_cta_max_hit()) {
    // Accept the stream operation if the maximum cycle/instruction/cta counts
    // are not triggered
    if (stream && !m_cuda_launch_blocking) {
      stream->push(op);
    } else {
      op.set_stream(&m_stream_zero);
      m_stream_zero.push(op);
    }
  } else {
    // Otherwise, ignore operation and continue
    printf(
        "GPGPU-Sim API: Maximum cycle, instruction, or CTA count hit. "
        "Skipping:");
    op.print(stdout);
    printf("\n");
  }
  if (g_debug_execution >= 3) print_impl(stdout);
  pthread_mutex_unlock(&m_lock);
  if (m_cuda_launch_blocking || stream == NULL) {
    unsigned int wait_amount = 100;
    unsigned int wait_cap = 100000;  // 100ms
    while (!empty()) {
      // sleep to prevent CPU hog by empty spin
      // sleep time increased exponentially ensure fast response when needed
      usleep(wait_amount);
      wait_amount *= 2;
      if (wait_amount > wait_cap) wait_amount = wait_cap;
    }
  }
}

void stream_manager::pushCudaStreamWaitEventToAllStreams(CUevent_st *e,
                                                         unsigned int flags) {
  std::list<CUstream_st *>::iterator s;
  for (s = m_streams.begin(); s != m_streams.end(); s++) {
    stream_operation op(*s, e, flags);
    push(op);
  }
}
