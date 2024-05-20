gazebo_gym_env_plugin/JointId joint_id
# Degrees of freedom of the joint, unused when using SetJointProperties
int32 degrees_of_freedom
# Lower position limit, when using in SetJointProperties keep the array empty to maintain the current configuration
float64[] position_limit_low
# Upper position limit, when using in SetJointProperties keep the array empty to maintain the current configuration
float64[] position_limit_high
