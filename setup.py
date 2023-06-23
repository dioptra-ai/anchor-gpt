from setuptools import setup, find_packages

import sys, os

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding='utf-8') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '\'' if '\'' in line else '\''
            return line.split(delim)[1]
        else:
            raise RuntimeError('Unable to find version string.')


if sys.version_info < (3, 5, 3):
    sys.exit('Sorry, Python < 3.5.3 is not supported')

long_description = read('README.md')
requirements = read('requirements.txt').splitlines()

__version__ = get_version('anchor_gpt/__init__.py')

setup(
    name='anchor-gpt',
    version=__version__,
    description='Diagnose and find hallucinations in your grounded Large Language Model prompts with Anchor-GPT!',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dioptra-ai/anchor-gpt',
    project_urls={
        'dioptra.ai': 'https://www.dioptra.ai',
    },
    author='dioptra.ai',
    author_email='info@dioptra.ai',
    license='BSD',
    python_requires='>=3.5.3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='dioptra, anchor, gpt, hallucinations, grounding, llm',
    packages=find_packages(exclude=['docs', 'tests*', 'examples']),
    include_package_data=True,
    install_requires=requirements
)