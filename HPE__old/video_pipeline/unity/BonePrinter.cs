using UnityEngine;

public class BonePrinter : MonoBehaviour
{
    void Start()
    {
        Animator anim = GetComponent<Animator>();
        if (anim == null) { Debug.LogError("No Animator found"); return; }

        foreach (HumanBodyBones bone in System.Enum.GetValues(typeof(HumanBodyBones)))
        {
            if (bone == HumanBodyBones.LastBone) continue;
            Transform t = anim.GetBoneTransform(bone);
            if (t != null)
                Debug.Log($"{bone} → {t.name} | localPos: {t.localPosition} | localRot: {t.localEulerAngles}");
            else
                Debug.Log($"{bone} → NOT MAPPED");
        }
    }
}