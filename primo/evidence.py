#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of PRIMO -- Probabilistic Inference Modules.
# Copyright (C) 2013-2015 Social Cognitive Systems Group, 
#                         Faculty of Technology, Bielefeld University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Lesser GNU General Public License as 
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public 
# License along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

class Evidence(object):
    '''
    A generic class for evidence. Can not be used on its own. Look for its
    subclasses.
    '''
    def __init__(self):
        pass

    def is_compatible(self, value):
        '''
        This method can be used to check if a value is consistent with some
        evidence.
        '''
        raise Exception("Not defined for this kind of Evidence")

    def get_unique_value(self):
        '''
        Sometimes only one value of some domain is compatible with the evidence.
        This is obviously the case for EvidenceEqual. It is then possible to
        use this value to speed up computations.

        @return: The only value compatible with the evidence or else None.
        '''
        return None

class EvidenceEqual(Evidence):
    '''
    This class can be used to specify evidence that a variable has taken some
    specified value.
    e.g. a=5
    '''
    def __init__(self, value):
        self.value=value

    def is_compatible(self, value):
        return self.value==value

    def get_unique_value(self):
        return self.value

class EvidenceInterval(Evidence):
    '''
    This class can be used to specify evidence that a variable has taken on
    some value in a defined interval.
    e.g. 2<=a<=5
    '''
    def __init__(self,min_val,max_val):
        self.min_val=min_val
        self.max_val=max_val

    def is_compatible(self, value):
        return self.min_val <= value and value<=self.max_val

    def get_interval(self):
        return self.min_val,self.max_val


class EvidenceLower(EvidenceInterval):
    '''
    This class can be used to specify evidence that a variable has taken on
    some value lower than some threshold.
    e.g. a<3
    '''
    def __init__(self,limit):
        super(EvidenceLower, self).__init__(float("-inf"),limit)


class EvidenceHigher(EvidenceInterval):
    '''
    This class can be used to specify evidence that a variable has taken on
    some value higher than some threshold.
    e.g. a>3
    '''
    def __init__(self,limit):
        super(EvidenceLower, self).__init__(limit,float("inf"))

