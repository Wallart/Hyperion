from setuptools import setup, find_packages


def sanitize_requirements():
    with open('requirements.txt') as f:
        install_reqs = f.read().splitlines()
    # Ignore comments and links from requirements.txt
    install_reqs = [r for r in install_reqs if '#' not in r and '+cu' not in r and '--find' not in r]

    pip_list, download_list = [], []
    for r in install_reqs:
        if 'git+' in r:
            download_list.append(r)
        else:
            pip_list.append(r)

    return pip_list, download_list


pip_list, download_list = sanitize_requirements()
found_packages = find_packages(where='.')
print(found_packages)
opts = {
    'author': 'Julien WALLART',
    'author_email': 'julien.wallart@outlook.com',
    'name': 'hyperion',
    'version': '1.3',
    'packages': found_packages,
    'install_requires': pip_list,
    'dependency_links': download_list,
    'package_data': {'hyperion': ['resources']},
    'include_package_data': True,
}
setup(**opts)
