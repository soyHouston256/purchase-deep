from shopassist.attributes import UNKNOWN, answer_attribute, extract_attributes
from shopassist.llm import MockLLM, parse_llm_json
from shopassist.tools import validate_args


def test_attributes_verifiable_or_unknown():
    p = {"title": "Auriculares Bluetooth 5.0 inalámbricos negros, 30 horas de batería", "brand": "Sony", "color": "",
         "description": "Cable USB-C de 1,2 m incluido. Peso 250 g. Garantía de 2 años", "bullet_point": "Silicona"}
    a = extract_attributes(p)
    assert a["marca"] == "Sony" and a["color"] == "negro" and a["peso"] == "250 g"
    assert "Bluetooth 5.0" in a["conectividad"] and "USB-C" in a["conectividad"]
    assert a["garantia"] == "2 años" and a["material"] == "silicona"
    assert a["capacidad"] == UNKNOWN and a["voltaje"] == UNKNOWN
    assert answer_attribute(p, "resistencia al agua") == UNKNOWN
    assert answer_attribute(p, "almacenamiento") == UNKNOWN
    assert answer_attribute(p, "peso") == "250 g"


def test_validate_args():
    schema = {"type": "object", "properties": {"qty": {"type": "integer", "minimum": 1, "maximum": 50},
                                               "q": {"type": "string"}}, "required": ["qty"]}
    assert validate_args(schema, {"qty": "3"}) == ({"qty": 3}, None)
    assert validate_args(schema, {"qty": 2.5})[1] is not None
    assert validate_args(schema, {"qty": 0})[1] is not None
    assert validate_args(schema, {"qty": 51})[1] is not None
    assert validate_args(schema, {})[1] is not None
    assert validate_args(schema, {"qty": True})[1] is not None


def test_parse_llm_json_robust():
    assert parse_llm_json('```json\n{"action": "view_cart", "args": {}}\n```') == {"action": "view_cart", "args": {}}
    assert parse_llm_json('Pienso... {"action": "respond", "message": "hola {x}"} extra')["message"] == "hola {x}"
    assert parse_llm_json("{'action': 'view_cart', 'args': {},}")["action"] == "view_cart"
    assert parse_llm_json('{action: "respond", message: "ok"}')["action"] == "respond"
    assert parse_llm_json('{"action": "search_products", "args": {"query": "mouse"}')["args"]["query"] == "mouse"
    assert parse_llm_json("sin json") is None


def test_mock_llm():
    m = MockLLM([{"action": "respond", "message": "hola"}])
    assert parse_llm_json(m.generate([]))["message"] == "hola"
