clean_step2:
\tpython -m etl.etl_claims
\tpython -m etl.etl_operational
\tpython -m etl.etl_compliance

validate_step2:
\tpython -m ml.schemas.run_validations

validate_features:
\tpython -m ml.validation.validate_features

test_features:
\tpytest -q ml/tests/test_features.py

explain_fraud:
\tpython -m ml.explain_fraud_shap

explain_ops:
\tpython -m ml.explain_ops_shap

dashboard:
\tstreamlit run ui/medguard_dashboard.py