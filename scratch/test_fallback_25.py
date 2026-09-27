import json
from context_store import ContextStore
from composer import LLMComposer

cs = ContextStore()
composer = LLMComposer()

with open('magicpin-ai-challenge/dataset/triggers_seed.json') as f:
    data = json.load(f)

print(f"Loaded {len(data['triggers'])} triggers.")

for trig in data['triggers']:
    m_id = trig.get('merchant_id')
    c_id = trig.get('customer_id')
    m = cs.get_merchant(m_id) if m_id else None
    c = cs.get_customer(c_id) if c_id else None
    cat_slug = m.get('category_slug') if m else trig.get('payload', {}).get('category')
    cat = cs.get_category(cat_slug) if cat_slug else None
    
    plan = composer.build_message_plan(m, cat, c, trig, {})
    res = composer._deterministic_compose(
        trigger_type=trig.get('kind'),
        merchant=m,
        category=cat,
        customer=c,
        trigger=trig,
        send_as='vera',
        template_name='t',
        template_params=[],
        suppression_key='supp'
    )
    print(f"=== {trig['id']} ({trig['kind']}) ===")
    print(f"BODY: {res['body']}")
    print(f"CTA: {res['cta']}")
    print()
