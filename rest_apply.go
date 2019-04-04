package graphpipe

import (
	"encoding/json"
)

type RESTApplier func(inputs map[string]json.RawMessage, outputNames []string) (interface{}, error)
