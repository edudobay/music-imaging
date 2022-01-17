from setuptools import setup

packages = ['imaging', 'imaging.core']

setup(
    name='imaging-tools',
    version='0.1',
    author='Eduardo Dobay',
    author_email='edudobay@gmail.com',
    packages=packages,
    install_requires=[
        'pillow',
    ],
    extras_require={
        'lines': ['scikit-image', 'numpy'],
    },
    entry_points={
        'console_scripts': [
            'imaging-join=imaging.scripts.join:main',
            'imaging-cut=imaging.scripts.cut:main',
            'imaging-lines=imaging.scripts.lines:main',
        ]
    },
)
