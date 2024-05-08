from setuptools import setup
import os
from glob import glob

package_name = 'situational_graphs_reasoning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),glob(os.path.join('config', '*.json'))),
        (os.path.join('share', package_name, 'pths'),glob(os.path.join('pths', '*.pth')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='josmilrom@gmail.com',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'situational_graphs_reasoning = situational_graphs_reasoning.situational_graphs_reasoning_node:main',
        ],
    },
)