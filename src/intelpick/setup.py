from glob import glob

from setuptools import find_packages, setup

package_name = 'intelpick'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artemis13',
    maintainer_email='adilsalkimbaev29@gmail.com',
    description='Camera + detector + RoArm pick-and-place sorting cell.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = intelpick.camera_node:main',
            'detector_node = intelpick.detector_node:main',
            'arm_node = intelpick.arm_node:main',
            'sorter_node = intelpick.sorter_node:main',
            'calibrate = intelpick.calibrate:main',
            'probe_arm = intelpick.probe_arm:main',
        ],
    },
)
