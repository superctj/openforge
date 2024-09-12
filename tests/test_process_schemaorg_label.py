from openforge.prepare_sotab_v2_for_mrf import process_schemaorg_label


def test_process_schemaorg_label():
    label1 = "CreativeWork/name"
    processed_label1 = process_schemaorg_label(label1)
    assert processed_label1 == "creative work name"

    label2 = "Person/name"
    processed_label2 = process_schemaorg_label(label2)
    assert processed_label2 == "person name"

    label3 = "Language"
    processed_label3 = process_schemaorg_label(label3)
    assert processed_label3 == "language"

    label4 = "DateTime"
    processed_label4 = process_schemaorg_label(label4)
    assert processed_label4 == "date time"

    label5 = "addressLocality"
    processed_label5 = process_schemaorg_label(label5)
    assert processed_label5 == "address locality"

    label6 = "DayOfWeek"
    processed_label6 = process_schemaorg_label(label6)
    assert processed_label6 == "day of week"

    label7 = "IdentifierAT"
    processed_label7 = process_schemaorg_label(label7)
    assert processed_label7 == "identifier at"

    label8 = "currency"
    processed_label8 = process_schemaorg_label(label8)
    assert processed_label8 == "currency"
