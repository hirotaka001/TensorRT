#pragma once

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

static inline int check_gpu_memory_usage(
	const std::string gpu_memory_log,
	const std::string gpu_checker_sh)
{
	int gpu_memory_usage = -1;
#ifndef _WIN32
	// check gpu memory usage
	char cmd[1000] = "";
	sprintf(cmd, "sh %s > %s", gpu_checker_sh.c_str(), gpu_memory_log.c_str());
	const int res = system(cmd);
	if (res != 0) {
		return gpu_memory_usage;
	}
	std::ifstream ifs(gpu_memory_log);
	std::string gpu_memory_usage_str;
	getline(ifs, gpu_memory_usage_str);
	gpu_memory_usage = std::atoi(gpu_memory_usage_str.c_str());
	remove(gpu_memory_log.c_str());
#endif
	return gpu_memory_usage;
}
