from setuptools import setup
from exp_runner.version import __version__

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='exp-runner',
    version=__version__,
    author='Sofya Lipnitskaya',
    author_email='lipnitskaya.sofya@gmail.com',
    description='Framework for data analysis and machine learning experiments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/slipnitskaya/exp-runner.git',
    packages=['exp_runner'],
    python_requires='>=3.6',
    install_requires=[
        'tqdm>=4.28.1',
        'numpy>=1.15.4',
        'scikit-learn>=0.20.1'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Topic :: Education :: Testing',
        'Topic :: Office/Business',
        'Topic :: Other/Nonlisted Topic',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Natural Language :: English',
        'Typing :: Typed'
    ],
    keywords=[
        'pipeline framework modelling model training testing classification regression clustering',
        'pipeline framework data analysis'
    ]
)
