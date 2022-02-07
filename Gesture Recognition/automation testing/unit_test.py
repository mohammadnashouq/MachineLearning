#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:59:55 2021

@author: darshit
"""
import unittest
import texttospeech_v0 as tts
import os
import autocorrect as ac

class Test(unittest.TestCase):

    def file_saved_removed(self):
        s,d = tts.tts("test note")
        self.assertEqual(s, True)
        self.assertEqual(d, False)

    def autocorrect(self):
        test_dict = {'whaat':'what', 'lovvee':'love','cami':'came','pictri':'picture'}
        result = ac.checker(test_dict)
        self.assertEqual(result,True)
        
if __name__ == '__main__':
    unittest.main()
