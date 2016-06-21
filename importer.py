# -*- coding: utf-8 -*-
import inspect


class Importer(object):
    def __init__(self, base_class, glob):
        self.base_class = base_class
        self.glob = glob
        self.classes = []

    def load_all_modules(self):
        import sys
        import glob
        import imp
        search_paths = glob.glob(self.glob)
        modules = []
        for i, path in enumerate(search_paths):
            sys.path.append(path)
            try:
                f_result = imp.find_module('runners', [path])
            except ImportError:
                continue
            module = imp.load_module('runners{}'.format(i), *f_result)
            modules.append(module)
            sys.path.pop()
        return modules

    def import_all_classes(self):
        self.classes = []
        for module in self.load_all_modules():
            classes = dict(inspect.getmembers(module, inspect.isclass)).values()
            filtered_classes = self.filter_parent(classes, self.base_class)
            self.classes.extend(filtered_classes)
        return self.classes

    def filter_parent(self, classes, parent_class):
        classes_to_return = [
            c for c in classes if (self.base_class == inspect.getmro(c)[1])
        ]
        return classes_to_return
