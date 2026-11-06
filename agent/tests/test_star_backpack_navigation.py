"""Focused navigation timing checks, not real-device or OCR evidence."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('star_backpack_navigation_test',Path(__file__).resolve().parents[1]/'utils/star_backpack_navigation.py')
nav=importlib.util.module_from_spec(spec)
spec.loader.exec_module(nav)

class Frame:
    def __init__(self,page):self.page=page
    def __getitem__(self,key):return SimpleNamespace(mean=lambda:255)

class Context:
    def __init__(self,wait_succeeds=True,render_after_wait=True):
        self.page='observatory';self.calls=[];self.tasker=SimpleNamespace(stopping=False)
        self.wait_succeeds=wait_succeeds;self.render_after_wait=render_after_wait
    def run_recognition(self,node,image):
        expected={'已在界面-观星':'observatory','星石本地OCR-已在背包':'backpack'}.get(node)
        return SimpleNamespace(hit=expected is not None and image.page==expected)
    def run_task(self,node,override):
        self.calls.append((node,override))
        if node=='星石本地OCR-点击背包入口':self.page='transition'
        elif node=='星石本地OCR-已在背包' and self.wait_succeeds and self.render_after_wait:self.page='backpack'
        return SimpleNamespace(status=SimpleNamespace(succeeded=node!='星石本地OCR-已在背包' or self.wait_succeeds))

class NavigationTest(unittest.TestCase):
    def run_entry(self,context):
        self.captures=[];self.records=[]
        def capture(label):self.captures.append(label);return Frame(context.page)
        return nav.prepare_star_backpack_entry(context,capture,self.records.append)
    def test_delayed_transition_waits_for_identity_without_second_click(self):
        context=Context();image,result=self.run_entry(context)
        self.assertEqual(image.page,'backpack');self.assertTrue(result['success'])
        self.assertEqual([x[0] for x in context.calls],['星石本地OCR-点击背包入口','星石本地OCR-已在背包'])
        settings=context.calls[1][1]['星石本地OCR-已在背包']
        self.assertEqual(settings['timeout'],6000);self.assertEqual(settings['post_delay'],0)
        self.assertEqual(settings['next'],[]);self.assertEqual(settings['on_error'],[])
    def test_identity_timeout_stops_before_accepting_a_backpack_frame(self):
        context=Context(wait_succeeds=False)
        with self.assertRaisesRegex(RuntimeError,'navigation failed'):self.run_entry(context)
        self.assertNotIn('entry-main-backpack',self.captures)
        self.assertEqual(len(context.calls),2);self.assertFalse(self.records[0]['success'])
    def test_final_image_still_requires_backpack_identity(self):
        context=Context(render_after_wait=False)
        with self.assertRaisesRegex(RuntimeError,'backpack identity missing'):self.run_entry(context)
        self.assertFalse(self.records[0]['success'])

class BackpackReturnContext(Context):
    def __init__(self,return_wait_succeeds=True,return_render=True):
        super().__init__();self.page='backpack';self.return_wait_succeeds=return_wait_succeeds;self.return_render=return_render
    def run_task(self,node,override):
        if node=='返回上级界面':self.calls.append((node,override));self.page='return_transition';return SimpleNamespace(status=SimpleNamespace(succeeded=True))
        if node=='已在界面-观星':
            self.calls.append((node,override))
            if self.return_wait_succeeds and self.return_render:self.page='observatory'
            return SimpleNamespace(status=SimpleNamespace(succeeded=self.return_wait_succeeds))
        return super().run_task(node,override)

class BackpackReturnTests(unittest.TestCase):
    run_entry=NavigationTest.run_entry
    # Dedicated cases use the same helper, but do not rerun inherited entry cases.
    def test_delayed_return_waits_for_observatory_before_new_screenshot(self):
        context=BackpackReturnContext();image,result=self.run_entry(context)
        self.assertTrue(result['success']);self.assertEqual(image.page,'backpack')
        self.assertEqual([x[0] for x in context.calls],['返回上级界面','已在界面-观星','星石本地OCR-点击背包入口','星石本地OCR-已在背包'])
        self.assertEqual(context.calls[1][1]['已在界面-观星']['timeout'],6000)
    def test_return_timeout_does_not_click_backpack_entry(self):
        context=BackpackReturnContext(return_wait_succeeds=False)
        with self.assertRaisesRegex(RuntimeError,'navigation failed'):self.run_entry(context)
        self.assertNotIn('entry-observatory',self.captures);self.assertNotIn('星石本地OCR-点击背包入口',[x[0] for x in context.calls])
    def test_return_wait_does_not_bypass_fresh_image_identity(self):
        context=BackpackReturnContext(return_render=False)
        with self.assertRaisesRegex(RuntimeError,'back did not reach observatory'):self.run_entry(context)
        self.assertFalse(self.records[0]['success'])

class HomeEntryTests(unittest.TestCase):
    def test_home_enters_observatory_then_backpack_once(self):
        class HomeContext(Context):
            def __init__(self):super().__init__();self.page='home'
            def run_recognition(self,node,image):
                if node=='已在界面-左半屏':return SimpleNamespace(hit=image.page=='home')
                return super().run_recognition(node,image)
            def run_task(self,node,override):
                if node=='进入界面-观星':
                    self.calls.append((node,override));self.page='observatory'
                    return SimpleNamespace(status=SimpleNamespace(succeeded=True))
                return super().run_task(node,override)
        context=HomeContext();image,result=nav.prepare_star_backpack_entry(context,lambda label:Frame(context.page),lambda result:None)
        self.assertEqual(image.page,'backpack');self.assertTrue(result['success'])
        self.assertEqual([x[0] for x in context.calls],['进入界面-观星','星石本地OCR-点击背包入口','星石本地OCR-已在背包'])
        self.assertNotIn('返回上级界面',[x[0] for x in context.calls])

if __name__=='__main__':unittest.main()
