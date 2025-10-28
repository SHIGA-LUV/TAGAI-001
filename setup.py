#!/usr/bin/env python3
"""
Setup script for AI MyTag DJ Assistant
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name='ai-mytag-dj-assistant',
    version='1.0.0',
    author='AI MyTag Team',
    author_email='contact@aimytag.com',
    description='Intelligent AI-powered tagging system for DJ track libraries',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/aimytag/ai-mytag-dj-assistant',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Multimedia :: Sound/Audio',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'audio': ['librosa>=0.10.0'],
        'spotify': ['spotipy>=2.22.0'],
        'web': ['flask>=2.3.0', 'flask-cors>=4.0.0'],
        'cloud': ['aiohttp>=3.8.0', 'aiofiles>=23.0.0'],
        'ml': ['scikit-learn>=1.3.0'],
        'all': [
            'librosa>=0.10.0',
            'spotipy>=2.22.0',
            'flask>=2.3.0',
            'flask-cors>=4.0.0',
            'aiohttp>=3.8.0',
            'aiofiles>=23.0.0',
            'scikit-learn>=1.3.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'aimytag=rekordbox_ai_tagger:main',
            'aimytag-gui=enhanced_realtime_tagger:main',
            'aimytag-batch=batch_processor:main',
            'aimytag-web=web_interface:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.json', '*.html', '*.css', '*.js'],
    },
)
