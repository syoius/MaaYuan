import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from custom.action.agent_info_collector import _AgentInfoReader

class ScanBoundaryTests(unittest.TestCase):
    def reader(self):
        r=_AgentInfoReader.__new__(_AgentInfoReader)
        r._ensure_running=Mock();r._sleep_checked=Mock()
        r.context=SimpleNamespace(run_recognition=Mock(return_value=SimpleNamespace(hit=True)))
        r._require_page=Mock(return_value=np.zeros((1280,720,3),np.uint8))
        r.screenshot=Mock(return_value=r._require_page.return_value)
        r._operator_arrow_visible=Mock(return_value=True)
        r.click=Mock();r._wait_for_text_change=Mock(return_value='下一位')
        return r

    def test_absent_arrow_requires_fresh_main_page_confirmation(self):
        r=self.reader();r._operator_arrow_visible.return_value=False
        fresh=np.ones((1280,720,3),np.uint8);r.screenshot.return_value=fresh
        self.assertFalse(r._move_operator('next',{'name':'末位'}))
        self.assertEqual(r._operator_arrow_visible.call_count,2)
        self.assertIs(r._operator_arrow_visible.call_args.args[0],fresh)
        r.context.run_recognition.assert_called_once();r.click.assert_not_called()

    def test_arrow_reappearing_on_second_frame_still_moves(self):
        r=self.reader();r._operator_arrow_visible.side_effect=[False,True]
        self.assertTrue(r._move_operator('next',{'name':'中间'}))
        r.click.assert_called_once()

    def test_changed_page_is_not_a_boundary(self):
        r=self.reader();r._operator_arrow_visible.return_value=False
        r.context.run_recognition.return_value.hit=False
        with self.assertRaisesRegex(RuntimeError,'不在主界面'):r._move_operator('next',{'name':'中间'})
        r.click.assert_not_called()

    def test_missing_screenshot_is_not_a_boundary(self):
        r=self.reader();r._operator_arrow_visible.return_value=False;r.screenshot.return_value=None
        with self.assertRaises(RuntimeError):r._move_operator('next',{'name':'中间'})

    def test_visible_arrow_with_failed_transition_is_failure(self):
        r=self.reader();r._wait_for_text_change.return_value=None
        with self.assertRaisesRegex(RuntimeError,'翻页未确认'):r._move_operator('next',{'name':'中间'})

    def test_arrow_recognition_failure_is_not_an_absent_arrow(self):
        r=self.reader();del r._operator_arrow_visible
        r.context.run_recognition.return_value=None
        with self.assertRaisesRegex(RuntimeError,'未能执行'):r._operator_arrow_visible(r.screenshot(),'next')
        r.context.run_recognition.return_value=SimpleNamespace(hit=False)
        self.assertFalse(r._operator_arrow_visible(r.screenshot(),'next'))

    def test_previous_arrow_uses_mirrored_image(self):
        r=self.reader();del r._operator_arrow_visible
        image=r.screenshot();image[:,0]=255
        self.assertTrue(r._operator_arrow_visible(image,'previous'))
        received=r.context.run_recognition.call_args.args[1]
        self.assertTrue(np.all(received[:,-1]==255));self.assertTrue(np.all(image[:,0]==255))

    def test_seek_first_continues_until_left_boundary(self):
        r=self.reader();r._read_main=Mock(side_effect=[{'name':x} for x in ('三','二','一')])
        r._move_operator=Mock(side_effect=[True,True,False]);r._seek_first_operator()
        self.assertEqual([c.args[1]['name'] for c in r._move_operator.call_args_list],['三','二','一'])
        self.assertTrue(all(c.args[0]=='previous' for c in r._move_operator.call_args_list))

if __name__=='__main__':unittest.main()
