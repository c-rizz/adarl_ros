#!/bin/bash
echo "Current governor is:"
grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_gov*
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo "Governor now is:"
grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_gov*