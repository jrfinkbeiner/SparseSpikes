import os
package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
lib_path = os.path.join(package_path, "lib")

default_build_path = os.path.join(package_path, "build")
build_path = os.environ.get("SSAX_BUILD_PATH", default_build_path)
if not os.path.isabs(build_path):
    build_path = os.path.join(os.getcwd(), build_path)

default_ssax_so_path = os.path.join(build_path, "ssax_shared_object.so")
ssax_so_path = os.environ.get("SSAX_SO_PATH", default_ssax_so_path)
if not os.path.isabs(ssax_so_path):
    ssax_so_path = os.path.join(os.getcwd(), build_path)