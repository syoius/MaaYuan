"""Actual image geometry plus synthetic presence combinations; real OCR is audited separately."""
import copy
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent'))
from utils.star_local_ocr_shadow import Bridge,PACKAGED_RUNTIME,read_image
from utils.star_experience_ocr import recognize_experience,experience_summary
from custom.action.star_inventory_sync import compact_inventory

FIXTURE=Path(__file__).parent/'fixtures/star_experience_presence'
KINDS=('orange','purple','white')
VALUES=dict(orange=23,purple=363,white=155)


class PresenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=tempfile.TemporaryDirectory();cls.addClassCleanup(directory.cleanup)
        cls.bridge=Bridge(PACKAGED_RUNTIME,Path(directory.name)/'bridge.log');cls.addClassCleanup(cls.bridge.close)
        cls.three=read_image(FIXTURE/'all-three.jpg')
        cls.two=read_image(FIXTURE/'purple-white.png')
        cls.three_geometry=cls.bridge.experience_geometry(cls.three)
        cls.two_geometry=cls.bridge.experience_geometry(cls.two)
        cls.simplified=read_image(FIXTURE/'simplified-three.png')
        cls.simplified_geometry=cls.bridge.experience_geometry(cls.simplified)
        cls.tiles={}
        for c in cls.three_geometry['candidates']:
            r=c['iconRect'];pad=4
            crop=cls.three[r['y']-pad:r['y']+r['height']+pad,r['x']-pad:r['x']+r['width']+pad]
            cls.tiles[c['quality']['kind']]=cv2.resize(crop,(104,104),interpolation=cv2.INTER_AREA)

    def synthetic(self,kinds,slots=None):
        """A synthetic arrangement of real icon patches, not real screenshot OCR evidence."""
        image=self.two.copy();bounds=self.two_geometry['bounds']
        y,h=bounds['y'],bounds['height'];w=image.shape[1]
        bg=np.median(image[y:y+h,540:720],axis=1).astype(np.uint8)
        image[y:y+h,:]=np.repeat(bg[:,None,:],w,axis=1)
        for i,kind in enumerate(kinds):
            x=round(w*(.165+.215*(i if slots is None else slots[i])))
            # Observed tab bottom and width-based gap, not a production fixed y.
            cy=round(self.two_geometry['tab']['rect']['y']+self.two_geometry['tab']['rect']['height']+w*.137)
            image[cy-52:cy+52,x-52:x+52]=self.tiles[kind]
        return image

    def provider(self,values,score=.99):
        source=iter(values);calls=[]
        def recognize(image):
            value=next(source);calls.append(image.shape)
            return dict(ocr_ms=1.,all_results=[dict(text=str(value),score=score,box=[0,0,image.shape[1],image.shape[0]])])
        return recognize,calls

    def test_original_real_images_have_correct_number_type_and_badge_regions(self):
        for image,geometry,kinds,points in [
            (self.three,self.three_geometry,KINDS,[(260,830),(530,830),(800,830)]),
            (self.two,self.two_geometry,('purple','white'),[(145,395),(300,395)]),
            (self.simplified,self.simplified_geometry,KINDS,[(150,395),(298,395),(459,395)])]:
            self.assertTrue(geometry['coverage']['complete'])
            self.assertEqual([c['quality']['kind'] for c in geometry['candidates']],list(kinds))
            bottom=geometry['tab']['rect']['y']+geometry['tab']['rect']['height']
            for candidate,(x,y) in zip(geometry['candidates'],points):
                self.assertGreater(candidate['iconRect']['y'],bottom)
                r=candidate['countRect']
                self.assertTrue(r['x']<=x<r['x']+r['width'] and r['y']<=y<r['y']+r['height'])
                self.assertLess(r['y']+r['height'],image.shape[0]*.5)

    def test_all_eight_presence_combinations_use_visual_kind_and_safe_absence(self):
        tested=[]
        for size in range(4):
            for present in itertools.combinations(KINDS,size):
                with self.subTest(present=present):
                    image=self.synthetic(present)
                    provider,calls=self.provider([VALUES[k] for k in present])
                    result=recognize_experience(image,provider,self.bridge)
                    self.assertTrue(result['complete'],result['warnings'])
                    self.assertEqual([i['kind'] for i in result['items']],list(present))
                    self.assertEqual([result[k+'Count'] for k in KINDS],[VALUES[k] if k in present else 0 for k in KINDS])
                    self.assertEqual(len(calls),size)
                    self.assertEqual(result['geometry']['coverage']['emptyConfirmed'],size==0)
                    tested.append(dict(present=present,counts={k:result[k+'Count'] for k in KINDS},complete=result['complete'],regions=[i['candidate'] for i in result['items']]))
        self.assertEqual(len(tested),8)
        type(self).combination_results=tested

    def test_search_and_count_regions_follow_observed_tab_translation(self):
        for dy in (12,24):
            image=np.roll(self.synthetic(('purple','white')),dy,axis=0)
            provider,calls=self.provider([VALUES['purple'],VALUES['white']])
            result=recognize_experience(image,provider,self.bridge)
            self.assertTrue(result['complete'],result['warnings'])
            self.assertGreater(result['geometry']['tab']['rect']['y'],self.two_geometry['tab']['rect']['y'])
            self.assertGreater(result['items'][0]['candidate']['iconRect']['y'],self.two_geometry['candidates'][0]['iconRect']['y'])
            self.assertEqual((result['orangeCount'],result['purpleCount'],result['whiteCount']),(0,363,155))
            self.assertEqual(len(calls),2)

    def test_present_item_bad_number_is_null_not_zero_and_retries_are_bounded(self):
        calls=[]
        def bad(image):
            calls.append(image.shape)
            return dict(ocr_ms=1.,all_results=[dict(text='bad',score=.99)])
        result=recognize_experience(self.two,bad,self.bridge)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['purpleCount']);self.assertIsNone(result['whiteCount'])
        self.assertLessEqual(len(calls),6)
        self.assertIn('experience_count_unconfirmed',result['warnings'])

    def test_unclassified_visible_icon_is_not_a_missing_kind(self):
        image=self.synthetic(('purple',))
        # Preserve the plate geometry but replace the gem's discriminative hue.
        image[329:382,92:146]=(0,180,0)
        provider,calls=self.provider([])
        result=recognize_experience(image,provider,self.bridge)
        self.assertFalse(result['complete']);self.assertEqual(calls,[])
        self.assertTrue(all(result[k+'Count'] is None for k in KINDS))

    def test_duplicate_kind_is_neither_added_nor_overwritten(self):
        image=self.synthetic(('purple','purple'))
        provider,calls=self.provider([4,7])
        result=recognize_experience(image,provider,self.bridge)
        self.assertFalse(result['complete']);self.assertIsNone(result['purpleCount'])
        self.assertIn('experience_kind_duplicate:purple',result['warnings'])
        self.assertEqual(len(calls),2)

    def test_unexplained_or_noncompacted_content_cannot_be_declared_empty(self):
        images=[self.synthetic(('white',),slots=[2]),self.synthetic(())]
        images[1][310:420,190:250]=(180,180,180)
        for image in images:
            provider,calls=self.provider([])
            result=recognize_experience(image,provider,self.bridge)
            self.assertFalse(result['complete']);self.assertEqual(calls,[])
            self.assertFalse(result['geometry']['coverage']['complete'])
            self.assertTrue(all(result[k+'Count'] is None for k in KINDS))

    def test_unselected_tab_and_header_currency_rois_never_run_count_ocr(self):
        image=self.synthetic(())
        tab=self.two_geometry['tab']['rect']
        image[tab['y']:tab['y']+tab['height'],tab['x']:tab['x']+tab['width']]=(45,43,65)
        provider,calls=self.provider([])
        result=recognize_experience(image,provider,self.bridge)
        self.assertFalse(result['complete']);self.assertEqual(calls,[])
        for mode in ('currency','tab'):
            geometry=copy.deepcopy(self.two_geometry)
            for c in geometry['candidates']:
                if mode=='currency':c['countRect']['y']=55
                else:c['iconRect']['y']=215
            fake=SimpleNamespace(experience_geometry=lambda im:geometry,experience_aggregate=self.bridge.experience_aggregate)
            provider,calls=self.provider([])
            result=recognize_experience(self.two,provider,fake)
            self.assertFalse(result['complete']);self.assertEqual(calls,[])
            self.assertIn('experience_count_outside_content' if mode=='currency' else 'experience_icon_outside_content',result['warnings'])

    def test_summary_and_compact_protocol_keep_names_and_exclude_experience_from_bag(self):
        for present in ((),('purple','white')):
            provider,_=self.provider([VALUES[k] for k in present])
            result=recognize_experience(self.synthetic(present),provider,self.bridge)
            summary=experience_summary(result)
            self.assertNotIn('items',summary);self.assertNotIn('geometry',summary)
            local=dict(schemaVersion=2,complete=True,sections={s:dict(complete=True,count=0,items=[]) for s in ('main','support')},
                experience=summary,bag=dict(complete=True,currentCount=0,capacity=290))
            payload=compact_inventory(local,'11111111-1111-4111-8111-111111111111')
            self.assertEqual(payload['counts'],dict(main=0,support=0))
            self.assertEqual(payload['bag'],dict(current_count=0,capacity=290))
            self.assertEqual(payload['experience'],{k:VALUES[k] if k in present else 0 for k in KINDS})
            self.assertEqual(set(payload),{'format','version','scan_id','complete','counts','inventory','experience','bag'})


if __name__=='__main__':unittest.main()
