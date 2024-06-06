from importlib.util import module_for_loader
from pkg_resources import require
from adarl.utils.utils import Pose, build_pose, buildRos1PoseStamped
import adarl.utils.utils

from typing import List, Dict, Optional
import os
import subprocess
from gazebo_msgs.srv import SpawnModelRequest,SpawnModel
import rospy
from gazebo_msgs.srv import DeleteModel, DeleteModelRequest
import adarl.utils.dbg.ggLog as ggLog
import time
from adarl.utils.utils import compile_xacro_string

spawned_models = []

serviceProxies = {}

def waitService(servicename, serviceclass):
    serviceProxy = serviceProxies.get(servicename, None)
    if serviceProxy is None:
        ggLog.info(f"gazbo_models_manager: Connecting to service {servicename}...")
        while True:
            try:
                rospy.wait_for_service(servicename, timeout=10)
                break
            except rospy.ROSException as e:
                ggLog.info(f"Waiting for service {servicename}... (got error {e})")
                time.sleep(1)
        serviceProxy = rospy.ServiceProxy(servicename, serviceclass)
        serviceProxies[servicename] = serviceProxy
    return serviceProxy


def compile_xacro(xacro_file_path : str, args : Dict[str,str]):
    args_str = []
    for k,v in args.items():
        args_str.append(f"{k}:={v} ")
    try:
        compiled_urdf = subprocess.check_output(["xacro", xacro_file_path]+args_str)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Xacro compilation failed with error. \n {e.stdout}\n {e.stderr} {adarl.utils.utils.exc_to_str(e)}")
    return compiled_urdf.decode("utf-8") 


def spawn_model(xacro_file_path : Optional[str] = None,
                pose : Pose = build_pose(0,0,0,0,0,0,1), 
                args : Dict[str,str] = {}, 
                model_name = "model", 
                robot_namespace = "", 
                reference_frame = "world",
                format = "urdf",
                xacro_string : Optional[str] = None):
    
    ggLog.info(f"Spawning model '{model_name}' from file {xacro_file_path} with args {args}")
    if xacro_string is not None and xacro_file_path is None:
        model_string = compile_xacro_string(model_definition_string=xacro_string,
                                            model_kwargs={})
    elif xacro_string is None and xacro_file_path is not None:
        model_string = compile_xacro(xacro_file_path,args)
    else:
        raise RuntimeError(f"Reeceived both xacro file and xacro string")
    ggLog.info(f"Compiled xacro is {model_string}")
    gazebo_namespace = "gazebo"
    if format == "urdf" or format == "urdf.xacro":
        spawn_model = waitService(gazebo_namespace+'/spawn_urdf_model', SpawnModel)
    elif format=="sdf" or format == "sdf.xacro":
        spawn_model = waitService(gazebo_namespace+'/spawn_sdf_model', SpawnModel)
    else:
        raise AttributeError(f"Unexpected format value '{format}'")


    while rospy.Time().now() == 0: # Wait for time to be non-zero (https://github.com/ros-simulation/gazebo_ros_pkgs/pull/1024)
        time.sleep(0.1)


    request = SpawnModelRequest()
    request.model_name = model_name
    request.model_xml = model_string
    request.robot_namespace = robot_namespace
    request.initial_pose = buildRos1PoseStamped(pose.position, pose.orientation_xyzw, frame_id=None).pose
    request.reference_frame = reference_frame
    
    response = spawn_model.call(request)

    if not response.success:
        raise RuntimeError(f"Failed to spawn model {xacro_file_path} with args {args}, response:\n  {response}")

    spawned_models.append(model_name)
    ggLog.info(f"Spawned model '{model_name}'")

def delete_model(model_name : str):
    ggLog.info(f"Deleting model '{model_name}'")
    request = DeleteModelRequest()
    request.model_name = model_name

    gazebo_namespace = "gazebo"
    delete_model = waitService(gazebo_namespace+'/delete_model', DeleteModel)
    response = delete_model.call(request)

    if not response.success:
        raise RuntimeError(f"Failed to delete model {model_name} response:\n  {response}")

    spawned_models.remove(model_name)
    ggLog.info(f"Deleted model '{model_name}'")

def delete_all_models():
    while len(spawned_models)>0:
        delete_model(spawned_models[-1])