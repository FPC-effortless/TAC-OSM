# Public publisher redirect amendment (no authentic readouts yet)

Original metadata-discovered path routes in694037e were rejected before body reads by the deliberately strict host/path guard. Run42259805778/artifact13524816016/digest258117c3cb776a13bb0c673c08d7b0ad0f48e732c038116d1927149a4f68e055 preserves that failure.

A subsequent request ataa11df84e5775694f43fd89a43d92af386140dae recorded the publisher Location metadata without following it: run42260234923/artifact13525480084/digestaeb3dc5be7a269c91adea4f03a0fb045753228f910db36e1146ab3f729973681. The official api.researchdata.se route redirects to HTTPS scania.open.care.snd.se, exact path /downloads/public/2024-34/3/data/train_operational_readouts.csv, with no query. Zero source bytes received.

Permit only this host with its exact paired prefix and the original two TRAIN filenames. No wildcard hosts, directories, versions, query values, archives or protected datasets. Both publisher SHA256 requirements stay unchanged. Eight local acquisition guard tests passed, including wrong host/path pair and protected filename rejection. The inherited auditor stays byte-unchanged. This append-only route correction changes neither labels nor targets nor training authority. Actual acquisition/audit outcome remains to be observed.
