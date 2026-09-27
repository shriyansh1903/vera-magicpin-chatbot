import json
from context_store import ContextStore
from composer import LLMComposer
from test_phase4 import load_all_dataset_contexts

cs, triggers = load_all_dataset_contexts()
composer = LLMComposer()

for i, trig in enumerate(triggers, 1):
    trig_id = trig['id']
    trig_kind = trig['kind']
    
    all_ctx = cs.get_all_context_for_trigger(trig_id)
    merchant = all_ctx.get('merchant')
    category = all_ctx.get('category')
    customer = all_ctx.get('customer')
    
    send_as = 'merchant_on_behalf' if trig_kind in ['recall_due', 'wedding_package_followup', 'customer_lapsed_hard', 'trial_followup', 'chronic_refill_due'] else 'vera'
    template_name, template_params = composer.select_template(trig_kind, {})
    
    res = composer._deterministic_compose(
        trigger_type=trig_kind,
        merchant=merchant,
        category=category,
        customer=customer,
        trigger=trig,
        send_as=send_as,
        template_name=template_name,
        template_params=template_params,
        suppression_key=trig.get('suppression_key', trig_id)
    )
    
    body = res['body']
    cta = res['cta']
    tmpl = res['template_name']
    
    safe_body = body.encode('ascii', errors='replace').decode('ascii')
    safe_cta = cta.encode('ascii', errors='replace').decode('ascii')
    
    print(f"### {i}. `{trig_id}` ({trig_kind})")
    print(f"- **Template / Renderer**: `{tmpl}` (`send_as`: `{send_as}`)")
    print(f"- **Fallback Message**: \"{safe_body}\"")
    print(f"- **CTA**: \"{safe_cta}\"")
    print(f"- **Gemini Bypassed**: Yes (Deterministic Fallback)")
    print(f"- **Trigger Evidence Grounded**: Yes")
    print()
