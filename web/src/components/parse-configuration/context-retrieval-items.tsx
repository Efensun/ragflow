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
                <div className="mt-2 mb-4">
                  <h4 className="text-base font-medium mb-2">
                    {t('contextPromptStructureTip')}
                  </h4>
                </div>

                <Form.Item
                  name={[
                    'parser_config',
                    'context_retrieval',
                    'document_context_prompt',
                  ]}
                  label={t('documentContextPrompt')}
                  tooltip={t('documentContextPromptTip')}
                  initialValue={DOCUMENT_CONTEXT_PROMPT}
                  extra={
                    <Tooltip
                      title={
                        documentCopied ? t('common.copied') : t('common.copy')
                      }
                    >
                      <Button
                        type="text"
                        size="small"
                        icon={
                          documentCopied ? <CheckOutlined /> : <CopyOutlined />
                        }
                        onClick={() =>
                          copyToClipboard(
                            DOCUMENT_CONTEXT_PROMPT.trim(),
                            setDocumentCopied,
                          )
                        }
                      />
                    </Tooltip>
                  }
                >
                  <Input.TextArea rows={4} />
                </Form.Item>

                <Form.Item
                  name={[
                    'parser_config',
                    'context_retrieval',
                    'chunk_context_prompt',
                  ]}
                  label={t('chunkContextPrompt')}
                  tooltip={t('chunkContextPromptTip')}
                  initialValue={CHUNK_CONTEXT_PROMPT}
                  extra={
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
                  }
                >
                  <Input.TextArea rows={8} />
                </Form.Item>
              </>
            )
          );
        }}
      </Form.Item>
    </DatasetConfigurationContainer>
  );
};

export default ContextRetrievalItems;
