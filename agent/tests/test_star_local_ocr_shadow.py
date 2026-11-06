import sys,unittest,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.star_local_ocr_shadow import pair_status,match_pair,read_image,REPO

class ShadowTests(unittest.TestCase):
    def test_conflicting_or_unknown_fields_cannot_merge(self):
        old={"name":"太阳","level":60,"quality":"橙"}
        for key,value in [("name","巨门"),("level",59),("quality","紫")]:
            self.assertEqual(pair_status(old,{**old,key:value},True)[0],"conflict")
        for key in old:self.assertEqual(pair_status(old,{**old,key:None},True)[0],"unconfirmed")
        self.assertEqual(pair_status(old,old,False)[0],"geometry_failed")
        self.assertEqual(pair_status(old,old,True)[0],"matched")
    def test_real_F1_correspondences_are_spatial_not_global_name_dedup(self):
        out=REPO/".ux/audits/star-local-ocr-live-20261031/offline-f1-complete"
        images=json.loads((out/"ocr-results.json").read_text(encoding="utf-8"))["images"]
        a,b=[copy.deepcopy(i) for i in images];im_a=read_image(out/"before.png");im_b=read_image(out/"after.png")
        result=match_pair(a,b,im_a,im_b)
        self.assertEqual((result["deduplicated_count"],result["different_position_candidate_count"]),(4,28))
        # Many other cards have the same name/level. They must remain separate candidates.
        target=next(x for x in b["records"] if x["row"]==0 and x["column"]==0)
        target["resolved"]["level"]=59
        conflict=match_pair(a,b,im_a,im_b)
        self.assertEqual((conflict["deduplicated_count"],conflict["different_position_candidate_count"],conflict["conflict_count"]),(3,29,1))
        self.assertTrue(conflict["no_business_instance_ids_generated"])

if __name__=="__main__":unittest.main()
