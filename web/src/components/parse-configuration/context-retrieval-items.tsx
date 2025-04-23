import { useTranslate } from '@/hooks/common-hooks';
import { cn } from '@/lib/utils';
import { CheckOutlined, CopyOutlined } from '@ant-design/icons';
import { Button, Form, Input, Switch, Tooltip } from 'antd';
import { useCallback, useState } from 'react';
import { DatasetConfigurationContainer } from '../dataset-configuration-container';

type ContextRetrievalItemsProps = {
  marginBottom?: boolean;
};

const DOCUMENT_CONTEXT_PROMPT = `
<document>
{doc_content}
</document>
`;

const CHUNK_CONTEXT_PROMPT = `
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
`;

export function UseContextRetrievalItem() {
  const { t } = useTranslate('knowledgeConfiguration');

  return (
    <Form.Item
      name={['parser_config', 'context_retrieval', 'use_context_retrieval']}
      label={t('useContextRetrieval')}
      initialValue={false}
      valuePropName="checked"
      tooltip={t('useContextRetrievalTip')}
    >
      <Switch />
    </Form.Item>
  );
}

const ContextRetrievalItems = ({
  marginBottom = false,
}: ContextRetrievalItemsProps) => {
  const { t } = useTranslate('knowledgeConfiguration');
  const [documentCopied, setDocumentCopied] = useState(false);
  const [chunkCopied, setChunkCopied] = useState(false);

  const renderWideTooltip = useCallback(
    (title: React.ReactNode | string) => {
      return {
        title: typeof title === 'string' ? t(title) : title,
        overlayInnerStyle: { width: '32vw' },
      };
    },
    [t],
  );

  const copyToClipboard = (
    text: string,
    setStateFn: (value: boolean) => void,
  ) => {
    navigator.clipboard.writeText(text);
    setStateFn(true);
    setTimeout(() => {
      setStateFn(false);
    }, 2000);
  };

  const DEFAULT_CONTEXT_PROMPT = `${DOCUMENT_CONTEXT_PROMPT}

${CHUNK_CONTEXT_PROMPT}`;

  return (
    <DatasetConfigurationContainer className={cn({ 'mb-4': marginBottom })}>
      <UseContextRetrievalItem></UseContextRetrievalItem>
      <Form.Item
        shouldUpdate={(prevValues, curValues) =>
          prevValues.parser_config.context_retrieval?.use_context_retrieval !==
          curValues.parser_config.context_retrieval?.use_context_retrieval
        }
      >
        {({ getFieldValue }) => {
          const useContextRetrieval = getFieldValue([
            'parser_config',
            'context_retrieval',
            'use_context_retrieval',
          ]);

          return (
            useContextRetrieval && (
              <>
                <Form.Item
                  name={[
                    'parser_config',
                    'context_retrieval',
                    'context_prompt',
                  ]}
                  label={t('contextPrompt')}
                  tooltip={renderWideTooltip('contextPromptStructureTip')}
                  initialValue={DEFAULT_CONTEXT_PROMPT}
                >
                  <Input.TextArea rows={12} />
                </Form.Item>
                <div className="mt-4 mb-6">
                  <div className="border p-3 rounded-md bg-gray-50">
                    <h4 className="text-base font-medium mb-2">
                      {t('contextPromptStructureTip')}
                    </h4>
                    <div className="mb-4">
                      <div className="font-medium mb-1 flex justify-between items-center">
                        <span>{t('documentContextPrompt')}:</span>
                        <Tooltip
                          title={
                            documentCopied
                              ? t('common.copied')
                              : t('common.copy')
                          }
                        >
                          <Button
                            type="text"
                            size="small"
                            icon={
                              documentCopied ? (
                                <CheckOutlined />
                              ) : (
                                <CopyOutlined />
                              )
                            }
                            onClick={() =>
                              copyToClipboard(
                                DOCUMENT_CONTEXT_PROMPT.trim(),
                                setDocumentCopied,
                              )
                            }
                          />
                        </Tooltip>
                      </div>
                      <pre className="p-2 bg-white border rounded text-sm whitespace-pre-wrap">
                        {DOCUMENT_CONTEXT_PROMPT}
                      </pre>
                    </div>
                    <div>
                      <div className="font-medium mb-1 flex justify-between items-center">
                        <span>{t('chunkContextPrompt')}:</span>
                        <Tooltip
                          title={
                            chunkCopied ? t('common.copied') : t('common.copy')
                          }
                        >
                          <Button
                            type="text"
                            size="small"
                            icon={
                              chunkCopied ? <CheckOutlined /> : <CopyOutlined />
                            }
                            onClick={() =>
                              copyToClipboard(
                                CHUNK_CONTEXT_PROMPT.trim(),
                                setChunkCopied,
                              )
                            }
                          />
                        </Tooltip>
                      </div>
                      <pre className="p-2 bg-white border rounded text-sm whitespace-pre-wrap">
                        {CHUNK_CONTEXT_PROMPT}
                      </pre>
                    </div>
                  </div>
                </div>
              </>
            )
          );
        }}
      </Form.Item>
    </DatasetConfigurationContainer>
  );
};

export default ContextRetrievalItems;
