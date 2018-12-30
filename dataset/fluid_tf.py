###############################################################################
#
#   Fluid Dataset
#   Copyright 2018 Moritz Becher
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###############################################################################


###############################################################################
# fluidTF classes
# ===============
# This file is generated from the fluidTF specification schema
# at https://github.com/mobec/fluidTF using quicktype
# https://github.com/quicktype/quicktype) with the command:
# quicktype -l python --python-version 3.7 -s schema fluidTF.schema.json
###############################################################################

# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = fluid_tf_from_dict(json.loads(json_string))

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, TypeVar, Callable, Type, cast
from enum import Enum


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class ArrayView:
    """A view into an array generally representing a subset of the array."""
    """The index of the array."""
    array: int
    extensions: Optional[Dict[str, Dict[str, Any]]]
    extras: Any
    """The length of the arrayView in elements."""
    length: int
    """The user-defined name of this object."""
    name: Optional[str]
    """The offset into the array in elements."""
    offset: Optional[int]

    @staticmethod
    def from_dict(obj: Any) -> 'ArrayView':
        assert isinstance(obj, dict)
        array = from_int(obj.get("array"))
        extensions = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], obj.get("extensions"))
        extras = obj.get("extras")
        length = from_int(obj.get("length"))
        name = from_union([from_str, from_none], obj.get("name"))
        offset = from_union([from_int, from_none], obj.get("offset"))
        return ArrayView(array, extensions, extras, length, name, offset)

    def to_dict(self) -> dict:
        result: dict = {}
        result["array"] = from_int(self.array)
        result["extensions"] = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], self.extensions)
        result["extras"] = self.extras
        result["length"] = from_int(self.length)
        result["name"] = from_union([from_str, from_none], self.name)
        result["offset"] = from_union([from_int, from_none], self.offset)
        return result


@dataclass
class Array:
    """An array points to a file containing fluid sim data"""
    extensions: Optional[Dict[str, Dict[str, Any]]]
    extras: Any
    """The user-defined name of this object."""
    name: Optional[str]
    """The uri of an .npz."""
    uri: str

    @staticmethod
    def from_dict(obj: Any) -> 'Array':
        assert isinstance(obj, dict)
        extensions = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], obj.get("extensions"))
        extras = obj.get("extras")
        name = from_union([from_str, from_none], obj.get("name"))
        uri = from_str(obj.get("uri"))
        return Array(extensions, extras, name, uri)

    def to_dict(self) -> dict:
        result: dict = {}
        result["extensions"] = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], self.extensions)
        result["extras"] = self.extras
        result["name"] = from_union([from_str, from_none], self.name)
        result["uri"] = from_str(self.uri)
        return result


class GridType(Enum):
    """The type of the grid, similar to Mantaflows grid definitions."""
    ANY = "Any"
    FLAG = "Flag"
    INT = "int"
    LEVELSET = "Levelset"
    MAC = "MAC"
    REAL = "Real"
    VEC3 = "Vec3"


@dataclass
class Grid:
    """Description of a grid"""
    array_view: Optional[int]
    """The number of elements in the grid."""
    count: Optional[int]
    extensions: Optional[Dict[str, Dict[str, Any]]]
    extras: Any
    """The type of the grid, similar to Mantaflows grid definitions."""
    grid_type: Optional[GridType]
    name: Any
    """The starting offset of the grid elements."""
    offset: Optional[int]

    @staticmethod
    def from_dict(obj: Any) -> 'Grid':
        assert isinstance(obj, dict)
        array_view = from_union([from_int, from_none], obj.get("arrayView"))
        count = from_union([from_int, from_none], obj.get("count"))
        extensions = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], obj.get("extensions"))
        extras = obj.get("extras")
        grid_type = from_union([GridType, from_none], obj.get("gridType"))
        name = obj.get("name")
        offset = from_union([from_int, from_none], obj.get("offset"))
        return Grid(array_view, count, extensions, extras, grid_type, name, offset)

    def to_dict(self) -> dict:
        result: dict = {}
        result["arrayView"] = from_union([from_int, from_none], self.array_view)
        result["count"] = from_union([from_int, from_none], self.count)
        result["extensions"] = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], self.extensions)
        result["extras"] = self.extras
        result["gridType"] = from_union([lambda x: to_enum(GridType, x), from_none], self.grid_type)
        result["name"] = self.name
        result["offset"] = from_union([from_int, from_none], self.offset)
        return result


@dataclass
class Scene:
    """The grids of one scene"""
    extensions: Optional[Dict[str, Dict[str, Any]]]
    extras: Any
    """The indices of each root node."""
    grids: Optional[List[Grid]]
    """The user-defined name of this object."""
    name: Optional[str]

    @staticmethod
    def from_dict(obj: Any) -> 'Scene':
        assert isinstance(obj, dict)
        extensions = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], obj.get("extensions"))
        extras = obj.get("extras")
        grids = from_union([lambda x: from_list(Grid.from_dict, x), from_none], obj.get("grids"))
        name = from_union([from_str, from_none], obj.get("name"))
        return Scene(extensions, extras, grids, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["extensions"] = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], self.extensions)
        result["extras"] = self.extras
        result["grids"] = from_union([lambda x: from_list(lambda x: to_class(Grid, x), x), from_none], self.grids)
        result["name"] = from_union([from_str, from_none], self.name)
        return result


@dataclass
class FluidTf:
    """The root object for a fluid dataset"""
    """Views into the raw fluid simulation ressources"""
    array_views: Optional[List[ArrayView]]
    """The raw fluid simulation ressources"""
    arrays: Optional[List[Array]]
    """The scenes used in this dataset"""
    scenes: Optional[List[Scene]]
    extensions: Optional[Dict[str, Dict[str, Any]]]
    extras: Any

    @staticmethod
    def from_dict(obj: Any) -> 'FluidTf':
        assert isinstance(obj, dict)
        array_views = from_union([lambda x: from_list(ArrayView.from_dict, x), from_none], obj.get("array_views"))
        arrays = from_union([lambda x: from_list(Array.from_dict, x), from_none], obj.get("arrays"))
        scenes = from_union([lambda x: from_list(Scene.from_dict, x), from_none], obj.get("scenes"))
        extensions = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], obj.get("extensions"))
        extras = obj.get("extras")
        return FluidTf(array_views, arrays, scenes, extensions, extras)

    def to_dict(self) -> dict:
        result: dict = {}
        result["array_views"] = from_union([lambda x: from_list(lambda x: to_class(ArrayView, x), x), from_none], self.array_views)
        result["arrays"] = from_union([lambda x: from_list(lambda x: to_class(Array, x), x), from_none], self.arrays)
        result["scenes"] = from_union([lambda x: from_list(lambda x: to_class(Scene, x), x), from_none], self.scenes)
        result["extensions"] = from_union([lambda x: from_dict(lambda x: from_dict(lambda x: x, x), x), from_none], self.extensions)
        result["extras"] = self.extras
        return result


def fluid_tf_from_dict(s: Any) -> FluidTf:
    return FluidTf.from_dict(s)


def fluid_tf_to_dict(x: FluidTf) -> Any:
    return to_class(FluidTf, x)
