from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'graph_reasoning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
        'graph_reasoning': ['config/*.json'],
        'graph_reasoning': ['pths/*.pth'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),glob(os.path.join('src/graph_reasoning/config', '*.json'))),
        (os.path.join('share', package_name, 'pths'),glob(os.path.join('src/graph_reasoning/pths', '*.pth')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='josmilrom@gmail.com',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'graph_reasoning = graph_reasoning.graph_reasoning_node:main'
        ],
    },
)
